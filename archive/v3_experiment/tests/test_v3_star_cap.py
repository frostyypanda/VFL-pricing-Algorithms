"""Unit tests for v3 star cap (recent-vs-historic decline)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.star_cap import apply_star_cap, STAR_PRICE_CAP


def test_declining_star_above_pop_mean_is_capped():
    # Given: historic 10 pts (above pop 7.5), recent 6 pts (ratio 0.6 < 0.85)
    # When: apply_star_cap runs with proposed price 13.0
    # Then: capped to STAR_PRICE_CAP (11.0)
    result = apply_star_cap(price=13.0, recent_avg=6.0, historic_avg=10.0,
                             pop_mean=7.5)
    assert result == STAR_PRICE_CAP


def test_stable_star_is_not_capped():
    # Given: historic 10 pts, recent 9.5 pts (ratio 0.95 ≥ 0.85)
    # When: apply_star_cap runs
    # Then: price unchanged
    result = apply_star_cap(price=13.0, recent_avg=9.5, historic_avg=10.0,
                             pop_mean=7.5)
    assert result == 13.0


def test_low_historic_player_below_pop_mean_not_capped_even_if_declining():
    # Given: historic 6 pts (below pop 7.5), recent 3 pts (ratio 0.5 < 0.85)
    # When: apply_star_cap runs
    # Then: price unchanged — star cap is for *stars*, not for already-cheap players
    result = apply_star_cap(price=8.0, recent_avg=3.0, historic_avg=6.0,
                             pop_mean=7.5)
    assert result == 8.0


def test_missing_recent_data_not_capped():
    # Given: no recent-stage data for the player
    # When: apply_star_cap runs with recent_avg=None
    # Then: price unchanged
    result = apply_star_cap(price=13.0, recent_avg=None, historic_avg=10.0,
                             pop_mean=7.5)
    assert result == 13.0


def test_price_below_cap_not_raised():
    # Given: declining star with proposed price 9.0 (below cap of 11.0)
    # When: apply_star_cap runs
    # Then: price unchanged at 9.0 (cap only lowers, never raises)
    result = apply_star_cap(price=9.0, recent_avg=4.0, historic_avg=10.0,
                             pop_mean=7.5)
    assert result == 9.0
