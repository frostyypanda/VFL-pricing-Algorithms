"""Tests for the data pipeline."""
import pytest
from v2.data_loader import (
    load_all_data, load_played, split_walk_forward,
    load_manual_prices, normalize_team,
)
from v2.constants import CHINA_TEAMS


def test_load_all_data_filters_china():
    df = load_all_data()
    for team in df["Team"].unique():
        assert team not in CHINA_TEAMS, f"China team {team} not filtered"


def test_load_all_data_has_expected_columns():
    df = load_all_data([2025])
    for col in ["Player", "Team", "Pts", "P?", "W/L", "Year", "Stage"]:
        assert col in df.columns, f"Missing column {col}"


def test_load_played_only_played_games():
    df = load_played([2025])
    assert (df["P?"] == 1).all()
    assert len(df) > 0


def test_split_walk_forward_no_leakage():
    df = load_all_data([2024, 2025])
    train, actual = split_walk_forward(df, "Stage 1", 2025)

    # Train should not contain any 2025 Stage 1 data
    train_s1 = train[(train["Year"] == 2025) & (train["Stage"] == "Stage 1")]
    assert len(train_s1) == 0, "Leakage: training data contains target stage"

    # Actual should only be 2025 Stage 1
    assert (actual["Year"] == 2025).all()
    assert (actual["Stage"] == "Stage 1").all()
    assert len(actual) > 0


def test_split_walk_forward_train_has_prior_data():
    df = load_all_data([2024, 2025])
    train, _ = split_walk_forward(df, "Stage 1", 2025)

    # Should contain 2024 data
    assert 2024 in train["Year"].values
    # Should contain 2025 Kickoff
    assert "Kickoff" in train[train["Year"] == 2025]["Stage"].values


def test_manual_prices_count_and_mean():
    mp = load_manual_prices()
    assert len(mp) == 180, f"Expected 180 players, got {len(mp)}"
    mean = mp["Stage1_Price"].mean()
    assert 9.0 <= mean <= 9.2, f"Mean price {mean:.3f} outside expected range"


def test_manual_prices_range():
    mp = load_manual_prices()
    assert mp["Stage1_Price"].min() >= 5.0
    assert mp["Stage1_Price"].max() <= 15.0


def test_normalize_team_aliases():
    assert normalize_team("LEVIATAN") == "LEVIATÁN"
    assert normalize_team("KRU Esports") == "KRÜ Esports"
    assert normalize_team("FNATIC") == "FNATIC"  # no change
