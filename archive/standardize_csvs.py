"""
Standardize all VFL CSVs (2024, 2025, 2026) to a consistent clean format.

Changes:
  - Name columns: Team, Player (instead of unnamed)
  - UTF-8 encoding
  - Consistent column set (add 6ks/7ks to 2024/2025 as 0)
  - Remove START rows from 2026
  - Clean junk values (#DIV/0!)
  - Integer types for count columns
"""
import pandas as pd
import os

DIR = os.path.dirname(__file__)

FINAL_COLUMNS = [
    "Team", "Player", "Stage", "Wk", "Game",
    "Pts", "T.Pts", "P.Pts", "PPM", "Adj.VP", "P?",
    "0k", "5k", "10k", "15k", "20k", "25k", "30k", "35k", "40k", "45k", "50k",
    "4ks", "5ks", "6ks", "7ks",
    "TOP3", "TOP2", "TOP1",
    "1.5R2", "1.75R2", "2.0R2",
    "PR Avg.", "W/L",
    "Game Start VP", "Game End VP",
]

INT_COLUMNS = [
    "Pts", "T.Pts", "P.Pts", "P?",
    "0k", "5k", "10k", "15k", "20k", "25k", "30k", "35k", "40k", "45k", "50k",
    "4ks", "5ks", "6ks", "7ks",
    "TOP3", "TOP2", "TOP1",
    "1.5R2", "1.75R2", "2.0R2",
]


def load_and_clean(filename, has_6k7k=False):
    path = os.path.join(DIR, filename)
    df = pd.read_csv(path, encoding="latin-1")

    # Rename first two unnamed columns
    cols = list(df.columns)
    cols[0] = "Team"
    cols[1] = "Player"
    df.columns = cols

    # Remove START rows (2026 has these)
    if "Game" in df.columns:
        df = df[df["Game"] != "START"].copy()

    # Add missing columns
    if not has_6k7k:
        # Insert 6ks, 7ks after 5ks
        idx = df.columns.get_loc("5ks") + 1
        df.insert(idx, "6ks", 0)
        df.insert(idx + 1, "7ks", 0)

    # Clean #DIV/0! and other junk in PR Avg.
    if "PR Avg." in df.columns:
        df["PR Avg."] = df["PR Avg."].replace({"#DIV/0!": ""})
        # Convert to numeric where possible, keep as float
        df["PR Avg."] = pd.to_numeric(df["PR Avg."], errors="coerce")

    # Ensure column order matches
    df = df[FINAL_COLUMNS]

    # Convert integer columns (fill NaN with 0 first)
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Clean float columns
    df["PPM"] = pd.to_numeric(df["PPM"], errors="coerce").fillna(0).round(2)
    df["Adj.VP"] = pd.to_numeric(df["Adj.VP"], errors="coerce")
    df["Game Start VP"] = pd.to_numeric(df["Game Start VP"], errors="coerce")
    df["Game End VP"] = pd.to_numeric(df["Game End VP"], errors="coerce")

    # Drop rows where Team and Player are both NaN
    df = df.dropna(subset=["Team", "Player"], how="all")

    return df


def main():
    files = [
        ("2024 VFL.csv", False),
        ("2025 VFL.csv", False),
        ("2026 VFL.csv", True),
    ]

    for filename, has_6k7k in files:
        print(f"Processing {filename}...")
        df = load_and_clean(filename, has_6k7k=has_6k7k)

        out_path = os.path.join(DIR, filename)
        df.to_csv(out_path, index=False, encoding="utf-8")

        played = df[df["P?"] == 1]
        print(f"  {len(df)} rows, {played['Player'].nunique()} players, {played['Team'].nunique()} teams")
        print(f"  Columns: {list(df.columns)}")
        print()

    print("All files standardized.")


if __name__ == "__main__":
    main()
