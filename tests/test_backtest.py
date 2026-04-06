"""Walk-forward backtests: predict a stage, score against actual results."""
import pytest
import numpy as np
from v2.data_loader import load_all_data, split_walk_forward
from v2.expected_points import calibrate, compute_expected_pts


def _run_backtest(target_stage, target_year):
    """Backtest: predict target stage from prior data, compare to actual."""
    all_data = load_all_data()
    train, actual = split_walk_forward(all_data, target_stage, target_year)

    if len(actual) < 30:
        pytest.skip(f"Not enough actual data for {target_year} {target_stage}")

    train_played = train[train["P?"] == 1]
    if len(train_played) < 100:
        pytest.skip(f"Not enough training data for {target_year} {target_stage}")

    cal = calibrate(all_data)

    # Build roster from actual data
    actual_avg = actual.groupby("Player")["Pts"].mean()
    roster = {}
    for player in actual["Player"].unique():
        grp = actual[actual["Player"] == player]
        team = grp["Team"].iloc[0]
        roster[player] = {"team": team, "region": "EMEA", "role": "D"}

    ep = compute_expected_pts(train_played, roster, cal, schedule_gws=False)

    # Compare BasePts to actual avg Pts
    pred_map = dict(zip(ep["Player"], ep["BasePts"]))
    common = set(pred_map.keys()) & set(actual_avg.index)
    if len(common) < 20:
        pytest.skip("Too few common players")

    pred = np.array([pred_map[p] for p in common])
    act = np.array([actual_avg[p] for p in common])

    corr = np.corrcoef(pred, act)[0, 1]
    mae = np.mean(np.abs(pred - act))

    return {
        "n": len(common),
        "corr": corr,
        "mae": mae,
        "pred_mean": pred.mean(),
        "act_mean": act.mean(),
    }


def test_backtest_2025_stage1():
    r = _run_backtest("Stage 1", 2025)
    print(f"\n2025 S1: n={r['n']}, corr={r['corr']:.3f}, MAE={r['mae']:.2f}")
    assert r["corr"] > 0.35, f"Correlation {r['corr']:.3f} too low"
    assert r["mae"] < 3.5, f"MAE {r['mae']:.2f} too high"


def test_backtest_2025_stage2():
    r = _run_backtest("Stage 2", 2025)
    print(f"\n2025 S2: n={r['n']}, corr={r['corr']:.3f}, MAE={r['mae']:.2f}")
    assert r["corr"] > 0.35, f"Correlation {r['corr']:.3f} too low"
    assert r["mae"] < 3.5, f"MAE {r['mae']:.2f} too high"
