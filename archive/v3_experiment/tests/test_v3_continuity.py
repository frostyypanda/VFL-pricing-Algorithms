"""Unit tests for v3 team-continuity."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.continuity import compute_team_continuity, continuity_adjusted_estimate


def test_player_who_stayed_on_same_team_has_continuity_one():
    # Given: player who played 10 prior games all on team A, now on team A
    # When: compute_team_continuity runs
    # Then: continuity == 1.0
    rows = [{"Player": "loyal", "Team": "A"}] * 10
    played = pd.DataFrame(rows)
    roster = {"loyal": {"team": "A", "region": "AMER", "role": "D"}}
    out = compute_team_continuity(played, roster)
    assert out["loyal"] == 1.0


def test_player_who_just_moved_has_zero_continuity():
    # Given: player who played 10 games on team A, now on team B
    # When: compute_team_continuity runs
    # Then: continuity == 0.0
    rows = [{"Player": "mover", "Team": "A"}] * 10
    played = pd.DataFrame(rows)
    roster = {"mover": {"team": "B", "region": "AMER", "role": "D"}}
    out = compute_team_continuity(played, roster)
    assert out["mover"] == 0.0


def test_partial_continuity_for_player_with_mixed_history():
    # Given: 3 games on A, 7 games on B, now on B
    # When: compute_team_continuity runs
    # Then: continuity == 0.7
    rows = ([{"Player": "mixed", "Team": "A"}] * 3 +
            [{"Player": "mixed", "Team": "B"}] * 7)
    played = pd.DataFrame(rows)
    roster = {"mixed": {"team": "B", "region": "AMER", "role": "D"}}
    out = compute_team_continuity(played, roster)
    assert abs(out["mixed"] - 0.7) < 1e-9


def test_high_continuity_no_adjustment():
    # Given: high continuity (≥0.7)
    # When: continuity_adjusted_estimate runs with personal=12, role=7
    # Then: result == personal (no shrinkage to role)
    assert continuity_adjusted_estimate(12.0, 7.0, 0.8) == 12.0


def test_zero_continuity_half_to_role():
    # Given: zero continuity
    # When: continuity_adjusted_estimate runs
    # Then: result is 50/50 personal + role
    result = continuity_adjusted_estimate(12.0, 7.0, 0.0)
    assert result == 0.5 * 12.0 + 0.5 * 7.0


def test_rookie_with_no_prior_games_has_continuity_one():
    # Given: player not present in played_df
    # When: compute_team_continuity runs
    # Then: continuity defaults to 1.0 (no penalty for true rookies)
    played = pd.DataFrame(columns=["Player", "Team"])
    roster = {"rookie": {"team": "X", "region": "AMER", "role": "D"}}
    out = compute_team_continuity(played, roster)
    assert out["rookie"] == 1.0
