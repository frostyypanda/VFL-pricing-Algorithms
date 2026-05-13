"""Unit tests for v3 team-form decay."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.team_form import compute_team_form, compute_team_form_ratio


def test_declining_team_form_ratio_is_below_one():
    # Given: team won 80% in 2024 Kickoff but only 20% in 2025 Stage 2
    # When: compute_team_form_ratio runs
    # Then: ratio (decayed / pooled) < 1.0 indicating decline
    rows = []
    for _ in range(8):
        rows.append({"Team": "Decline", "Year": 2024, "Stage": "Kickoff", "W/L": "W"})
    for _ in range(2):
        rows.append({"Team": "Decline", "Year": 2024, "Stage": "Kickoff", "W/L": "L"})
    for _ in range(2):
        rows.append({"Team": "Decline", "Year": 2025, "Stage": "Stage 2", "W/L": "W"})
    for _ in range(8):
        rows.append({"Team": "Decline", "Year": 2025, "Stage": "Stage 2", "W/L": "L"})
    played = pd.DataFrame(rows)
    ratio = compute_team_form_ratio(played)
    assert ratio["Decline"] < 1.0


def test_improving_team_form_ratio_is_above_one():
    # Given: team won 20% in 2024 Kickoff and 80% in 2025 Stage 2
    # When: compute_team_form_ratio runs
    # Then: ratio > 1.0
    rows = []
    for _ in range(2):
        rows.append({"Team": "Rising", "Year": 2024, "Stage": "Kickoff", "W/L": "W"})
    for _ in range(8):
        rows.append({"Team": "Rising", "Year": 2024, "Stage": "Kickoff", "W/L": "L"})
    for _ in range(8):
        rows.append({"Team": "Rising", "Year": 2025, "Stage": "Stage 2", "W/L": "W"})
    for _ in range(2):
        rows.append({"Team": "Rising", "Year": 2025, "Stage": "Stage 2", "W/L": "L"})
    played = pd.DataFrame(rows)
    ratio = compute_team_form_ratio(played)
    assert ratio["Rising"] > 1.0


def test_stable_team_form_ratio_is_one():
    # Given: team won 50% in every stage
    # When: compute_team_form_ratio runs
    # Then: ratio == 1.0 (stable)
    rows = []
    for stage_year in [(2024, "Kickoff"), (2025, "Stage 2")]:
        for _ in range(5):
            rows.append({"Team": "Stable", "Year": stage_year[0],
                         "Stage": stage_year[1], "W/L": "W"})
        for _ in range(5):
            rows.append({"Team": "Stable", "Year": stage_year[0],
                         "Stage": stage_year[1], "W/L": "L"})
    played = pd.DataFrame(rows)
    ratio = compute_team_form_ratio(played)
    assert abs(ratio["Stable"] - 1.0) < 1e-9


def test_team_form_is_clamped_within_30_pct():
    # Given: a wildly improving team (0% old, 100% new)
    # When: compute_team_form_ratio runs
    # Then: ratio clamped to [0.7, 1.3]
    rows = []
    for _ in range(10):
        rows.append({"Team": "Wild", "Year": 2024, "Stage": "Kickoff", "W/L": "L"})
    for _ in range(10):
        rows.append({"Team": "Wild", "Year": 2025, "Stage": "Stage 2", "W/L": "W"})
    played = pd.DataFrame(rows)
    ratio = compute_team_form_ratio(played)
    assert 0.7 <= ratio["Wild"] <= 1.3
