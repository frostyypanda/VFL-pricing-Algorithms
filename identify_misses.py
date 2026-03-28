"""
Identify players where the algorithm diverges significantly (>=2.0 VP)
from manual prices, and diagnose WHY — focusing on data availability.

Usage:
    python identify_misses.py
"""
import pandas as pd
import numpy as np
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
#  China teams to filter out (same list as pricing_algorithms.py)
# ---------------------------------------------------------------------------
CHINA_TEAMS = {
    "Bilibili Gaming", "Dragon Ranger Gaming", "Edward Gaming",
    "FunPlus Phoenix", "JD Mall JDG Esports", "Nova Esports",
    "Titan Esports Club", "Trace Esports", "Wolves Esports",
    "Xi Lai Gaming", "TYLOO", "All Gamers",
}

# Training events used by vfl_2026_stage1.py
TRAINING_EVENTS = [
    (2024, "Kickoff"),
    (2024, "Madrid"),
    (2024, "Stage 1"),
    (2024, "Shanghai"),
    (2024, "Stage 2"),
    (2024, "Champions"),
    (2025, "Kickoff"),
    (2025, "bangkok"),
    (2025, "Stage 1"),
    (2025, "Toronto"),
    (2025, "Stage 2"),
    (2025, "Champions"),
    (2026, "Kickoff"),
    (2026, "Santiago"),
]

EVENTS_2026 = [(2026, "Kickoff"), (2026, "Santiago")]


# ---------------------------------------------------------------------------
#  Team name normalization (mirrors vfl_2026_stage1.py)
# ---------------------------------------------------------------------------
def _normalize_team(name):
    if not isinstance(name, str):
        return name
    for old, new in [
        ("KR\x9a Esports", "KRU Esports"),
        ("KRÜ Esports",    "KRU Esports"),
        ("KR\xdc Esports", "KRU Esports"),
        ("LEVIAT\xb5N",    "LEVIATAN"),
        ("LEVIATÁN",       "LEVIATAN"),
        ("LEVIAT\xc1N",    "LEVIATAN"),
        ("PCIFIC Espor",   "PCIFIC Esports"),
        ("ULF Esports",    "Eternal Fire"),
    ]:
        if name == old:
            return new
    return name


def _read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


# ---------------------------------------------------------------------------
#  Load data
# ---------------------------------------------------------------------------
def load_all():
    frames = []
    for year, fn in [(2024, "2024 VFL.csv"), (2025, "2025 VFL.csv"),
                     (2026, "2026 VFL.csv")]:
        df = _read_csv(os.path.join(DIR, fn))
        if "Team" not in df.columns:
            cols = list(df.columns)
            if cols[0].startswith("Unnamed") and cols[1].startswith("Unnamed"):
                cols[0], cols[1] = "Team", "Player"
                df.columns = cols
        df["Year"] = year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Team"] = combined["Team"].apply(_normalize_team)
    combined = combined[~combined["Team"].isin(CHINA_TEAMS)].copy()
    return combined


def load_generated_prices():
    path = os.path.join(DIR, "generated_prices_2026.csv")
    df = _read_csv(path)
    df["Team"] = df["Team"].apply(_normalize_team)
    return df


def load_manual_prices():
    path = os.path.join(DIR, "manual_prices_2026.csv")
    df = _read_csv(path)
    df["Team"] = df["Team"].apply(_normalize_team)
    return df


# ---------------------------------------------------------------------------
#  Count games per player across training events
# ---------------------------------------------------------------------------
def count_games(all_data):
    """Return a single DataFrame with games per player: total, per year, 2026-only."""
    played = all_data[all_data["P?"] == 1].copy()
    played["Player_lower"] = played["Player"].str.lower()

    # Total training games
    total = played.groupby("Player_lower").size().reset_index(name="games_total")

    # Per-year counts
    for y in [2024, 2025, 2026]:
        mask = played["Year"] == y
        ydf = played[mask].groupby("Player_lower").size().reset_index(name=f"games_{y}")
        total = total.merge(ydf, on="Player_lower", how="left")

    for y in [2024, 2025, 2026]:
        total[f"games_{y}"] = total[f"games_{y}"].fillna(0).astype(int)

    return total


# ---------------------------------------------------------------------------
#  Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  IDENTIFYING ALGORITHM MISSES (|diff| >= 2.0 VP)")
    print("=" * 80)

    all_data = load_all()
    manual_df = load_manual_prices()
    gen_df = load_generated_prices()

    game_counts = count_games(all_data)

    # Merge generated and manual prices
    manual_df["Player_lower"] = manual_df["Player"].str.lower()
    gen_df["Player_lower"] = gen_df["Player"].str.lower() if "Player" in gen_df.columns else gen_df.iloc[:, 0].str.lower()

    # Figure out generated price column name
    gen_price_col = None
    for col in gen_df.columns:
        if "vp" in col.lower() or "price" in col.lower() or "generated" in col.lower():
            gen_price_col = col
            break
    if gen_price_col is None:
        # Try numeric columns
        for col in gen_df.columns:
            if gen_df[col].dtype in [np.float64, np.int64]:
                gen_price_col = col
                break

    print(f"\n  Generated prices column: {gen_price_col}")
    print(f"  Generated prices shape: {gen_df.shape}")
    print(f"  Manual prices shape: {manual_df.shape}")

    # Build comparison DataFrame
    comparison = manual_df[["Player", "Team", "Position", "Price"]].copy()
    comparison.rename(columns={"Price": "manual_vp"}, inplace=True)
    comparison["Player_lower"] = comparison["Player"].str.lower()

    # Match generated prices
    gen_map = dict(zip(gen_df["Player_lower"], gen_df[gen_price_col]))
    comparison["generated_vp"] = comparison["Player_lower"].map(gen_map)

    # If generated_prices CSV doesn't exist or has issues, try parsing from output
    if comparison["generated_vp"].isna().all():
        print("\n  WARNING: Could not match generated prices from CSV. Check file format.")
        return

    comparison["diff"] = comparison["generated_vp"] - comparison["manual_vp"]
    comparison["abs_diff"] = comparison["diff"].abs()

    # Merge game counts
    comparison = comparison.merge(game_counts, on="Player_lower", how="left")
    for col in ["games_total", "games_2024", "games_2025", "games_2026"]:
        comparison[col] = comparison[col].fillna(0).astype(int)

    # Filter to >= 2.0 VP disagreements
    misses = comparison[comparison["abs_diff"] >= 2.0].sort_values("diff", ascending=False).copy()

    # Split into categories
    overpriced = misses[misses["diff"] >= 2.0].sort_values("diff", ascending=False)
    underpriced = misses[misses["diff"] <= -2.0].sort_values("diff", ascending=True)

    # Categorize data confidence
    def data_tag(row):
        tags = []
        if row["games_2026"] == 0:
            tags.append("NO 2026 data")
        elif row["games_2026"] <= 2:
            tags.append(f"only {row['games_2026']} games in 2026")
        if row["games_total"] <= 3:
            tags.append("very limited total data")
        elif row["games_total"] <= 6:
            tags.append("limited total data")
        g2024 = row.get("games_2024", 0)
        g2025 = row.get("games_2025", 0)
        g2026 = row.get("games_2026", 0)
        if g2024 > 0 and g2025 == 0 and g2026 == 0:
            tags.append("only 2024 data (stale)")
        if g2024 == 0 and g2025 == 0 and g2026 > 0:
            tags.append("brand new player (2026 only)")
        if g2024 == 0 and g2025 > 0 and g2026 == 0:
            tags.append("no 2026 data, only 2025")
        return "; ".join(tags) if tags else "adequate data"

    misses["data_notes"] = misses.apply(data_tag, axis=1)
    overpriced = overpriced.copy()
    overpriced["data_notes"] = overpriced.apply(data_tag, axis=1)
    underpriced = underpriced.copy()
    underpriced["data_notes"] = underpriced.apply(data_tag, axis=1)

    # ---------------------------------------------------------------------------
    #  Print results
    # ---------------------------------------------------------------------------
    def print_section(title, df, direction):
        print(f"\n{'=' * 80}")
        print(f"  {title} ({len(df)} players)")
        print(f"{'=' * 80}")
        print(f"  {'Player':<20} {'Team':<22} {'Pos':>3}  {'Gen':>5}  {'Man':>5}  {'Diff':>6}  "
              f"{'Tot':>4} {'2024':>4} {'2025':>4} {'2026':>4}  Data Notes")
        print(f"  {'-'*130}")
        for _, row in df.iterrows():
            g2024 = row.get("games_2024", 0)
            g2025 = row.get("games_2025", 0)
            g2026 = row.get("games_2026", 0)
            diff_str = f"+{row['diff']:.1f}" if row['diff'] > 0 else f"{row['diff']:.1f}"
            notes = row.get("data_notes", "")
            print(f"  {row['Player']:<20} {row['Team']:<22} {row['Position']:>3}  "
                  f"{row['generated_vp']:>5.1f}  {row['manual_vp']:>5.1f}  {diff_str:>6}  "
                  f"{row['games_total']:>4} {g2024:>4} {g2025:>4} {g2026:>4}  {notes}")

    print_section("ALGO OVERPRICED (algo > manual by 2+ VP) — needs manual price adjustment DOWN?",
                  overpriced, "over")
    print_section("ALGO UNDERPRICED (algo < manual by 2+ VP) — needs manual price adjustment UP?",
                  underpriced, "under")

    # ---------------------------------------------------------------------------
    #  Summary statistics
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total players priced:          {len(comparison)}")
    print(f"  Players with |diff| >= 2.0 VP: {len(misses)} ({100*len(misses)/len(comparison):.1f}%)")
    print(f"  Algo overpriced (>= +2.0):     {len(overpriced)}")
    print(f"  Algo underpriced (<= -2.0):    {len(underpriced)}")
    print()

    # Data confidence breakdown for misses
    low_data = misses[misses["games_2026"] <= 2]
    no_data = misses[misses["games_2026"] == 0]
    print(f"  Of the {len(misses)} misses:")
    print(f"    {len(no_data)} have NO 2026 games at all")
    print(f"    {len(low_data)} have <= 2 games in 2026")
    print(f"    {len(misses[misses['games_total'] <= 6])} have <= 6 total training games")
    print()

    # Biggest misses by category
    print(f"  TOP 5 OVERPRICED (algo thinks better than they are):")
    for _, r in overpriced.head(5).iterrows():
        print(f"    {r['Player']:<18} {r['Team']:<22} gen={r['generated_vp']:.1f} man={r['manual_vp']:.1f} "
              f"(+{r['diff']:.1f})  2026 games: {r['games_2026']}")
    print(f"\n  TOP 5 UNDERPRICED (algo thinks worse than they are):")
    for _, r in underpriced.head(5).iterrows():
        print(f"    {r['Player']:<18} {r['Team']:<22} gen={r['generated_vp']:.1f} man={r['manual_vp']:.1f} "
              f"({r['diff']:.1f})  2026 games: {r['games_2026']}")

    # ---------------------------------------------------------------------------
    #  Needs user input: group by likely cause
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  NEEDS USER INPUT — GROUPED BY LIKELY CAUSE")
    print(f"{'=' * 80}")

    # 1. New/role-changed players with very limited data
    limited = misses[misses["games_2026"] <= 2].sort_values("abs_diff", ascending=False)
    if len(limited) > 0:
        print(f"\n  [A] LIMITED 2026 DATA (<= 2 games) — algo extrapolating from old data")
        print(f"  {'Player':<20} {'Team':<22} {'Pos':>3}  {'Gen':>5}  {'Man':>5}  {'Diff':>6}  "
              f"{'2026':>4} {'Total':>5}  Direction")
        print(f"  {'-'*110}")
        for _, r in limited.iterrows():
            diff_str = f"+{r['diff']:.1f}" if r['diff'] > 0 else f"{r['diff']:.1f}"
            direction = "OVER" if r['diff'] > 0 else "UNDER"
            print(f"  {r['Player']:<20} {r['Team']:<22} {r['Position']:>3}  "
                  f"{r['generated_vp']:>5.1f}  {r['manual_vp']:>5.1f}  {diff_str:>6}  "
                  f"{r['games_2026']:>4} {r['games_total']:>5}  {direction}")

    # 2. Players with adequate data but still big disagreement
    adequate = misses[misses["games_2026"] > 2].sort_values("abs_diff", ascending=False)
    if len(adequate) > 0:
        print(f"\n  [B] ADEQUATE 2026 DATA (>2 games) — genuine algo vs human disagreement")
        print(f"  {'Player':<20} {'Team':<22} {'Pos':>3}  {'Gen':>5}  {'Man':>5}  {'Diff':>6}  "
              f"{'2026':>4} {'Total':>5}  Direction")
        print(f"  {'-'*110}")
        for _, r in adequate.iterrows():
            diff_str = f"+{r['diff']:.1f}" if r['diff'] > 0 else f"{r['diff']:.1f}"
            direction = "OVER" if r['diff'] > 0 else "UNDER"
            print(f"  {r['Player']:<20} {r['Team']:<22} {r['Position']:>3}  "
                  f"{r['generated_vp']:>5.1f}  {r['manual_vp']:>5.1f}  {diff_str:>6}  "
                  f"{r['games_2026']:>4} {r['games_total']:>5}  {direction}")

    print()


if __name__ == "__main__":
    main()
