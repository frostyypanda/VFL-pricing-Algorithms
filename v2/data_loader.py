"""Data loading and standardization for VFL v2."""
import os
import pandas as pd
import numpy as np
from .constants import CHINA_TEAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data")

TEAM_ALIASES = {
    "ULF Esports": "Eternal Fire",
    "PCIFIC Espor": "PCIFIC Esports",
    "LEVIATµN": "LEVIATÁN", "Leviatan": "LEVIATÁN",
    "LEVIATAN": "LEVIATÁN", "Leviatán": "LEVIATÁN",
    "KR Esports": "KRÜ Esports", "KRU Esports": "KRÜ Esports",
    "2Game Esports": "ENVY", "The Guard": "ENVY",
}

# Stage ordering for walk-forward splits
STAGE_ORDER = [
    "Kickoff", "bangkok", "Madrid", "Santiago",
    "Stage 1", "Toronto", "Shanghai", "London",
    "Stage 2", "Champions",
]


def normalize_team(name):
    if not isinstance(name, str):
        return str(name)
    return TEAM_ALIASES.get(name.strip(), name.strip())


def load_vfl_csv(path, year):
    """Load a single VFL CSV and add Year column."""
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    cols = list(df.columns)
    if "Unnamed: 0" in cols:
        df = df.rename(columns={cols[0]: "Team", cols[1]: "Player"})

    for col in ["6ks", "7ks"]:
        if col not in df.columns:
            df[col] = 0

    df["Year"] = year
    df["Team"] = df["Team"].apply(normalize_team)
    df["P?"] = pd.to_numeric(df["P?"], errors="coerce").fillna(0).astype(int)

    num_cols = [
        "Pts", "T.Pts", "P.Pts", "PPM",
        "4ks", "5ks", "6ks", "7ks",
        "TOP3", "TOP2", "TOP1", "1.5R2", "1.75R2", "2.0R2",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def load_all_data(years=None):
    """Load and concatenate VFL CSVs. Filter China."""
    if years is None:
        years = [2024, 2025, 2026]
    frames = []
    for year in years:
        path = os.path.join(DIR, f"{year} VFL.csv")
        if os.path.exists(path):
            frames.append(load_vfl_csv(path, year))
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["Team"].isin(CHINA_TEAMS)]
    return df


def load_played(years=None):
    """Load only played games (P?=1)."""
    return load_all_data(years).query("`P?` == 1").copy()


def split_walk_forward(df, target_stage, target_year):
    """Split data into train (before target) and actual (the target).

    Returns (train_played, actual_played).
    """
    stage_idx = {s: i for i, s in enumerate(STAGE_ORDER)}
    target_i = stage_idx.get(target_stage, 99)

    def is_before(row):
        if row["Year"] < target_year:
            return True
        if row["Year"] == target_year:
            return stage_idx.get(row["Stage"], 99) < target_i
        return False

    def is_target(row):
        return row["Year"] == target_year and row["Stage"] == target_stage

    mask_before = df.apply(is_before, axis=1)
    mask_target = df.apply(is_target, axis=1)

    train = df[mask_before & (df["P?"] == 1)].copy()
    actual = df[mask_target & (df["P?"] == 1)].copy()
    return train, actual


def load_manual_prices():
    """Load the 2026 Stage 1 manual pricing sheet."""
    path = os.path.join(
        DIR, "Stats for VFL Prices 2026 - Stage 1 VFL Pricing Sheet.csv"
    )
    raw = pd.read_csv(path, encoding="utf-8-sig", header=None)

    headers = [
        "Player", "Team", "Region", "Position_Full", "Position",
        "Stage1_Price", "Santiago_Price", "Santiago_WkAvg",
        "Santiago_PPG", "Santiago_VLR_R", "Santiago_KPR",
        "Kickoff_Price", "Kickoff_3Game", "Kickoff_PPG",
        "Kickoff_VLR_R", "Kickoff_KPR",
    ]

    rows = []
    for i in range(6, len(raw)):  # data starts row 6
        player = raw.iloc[i, 1]
        if not isinstance(player, str) or player.strip() == "":
            continue
        entry = {}
        for j, h in enumerate(headers):
            val = raw.iloc[i, j + 1] if (j + 1) < len(raw.columns) else None
            entry[h] = val
        rows.append(entry)

    df = pd.DataFrame(rows)
    df["Team"] = df["Team"].apply(normalize_team)
    for col in ["Stage1_Price", "Santiago_Price", "Kickoff_Price",
                 "Santiago_PPG", "Santiago_VLR_R", "Santiago_KPR",
                 "Kickoff_PPG", "Kickoff_VLR_R", "Kickoff_KPR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pickrate_summary():
    """Load pickrate summary data."""
    path = os.path.join(DIR, "pickrate_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)
