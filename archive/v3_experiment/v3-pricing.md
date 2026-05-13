# Feature: v3 Pricing Algorithm

## Status
deprecated (2026-05-14)

> v3-full and each component individually underperform v2 across an 8-holdout backtest. See `output/v3_findings.md`. Code retained for ablation/research; not enabled by default.

## Why this exists

After the 2026 Stage 1 backtest (see `output/Stage1_2026_pricing_learnings.md`), three failure modes were identified that affect Algo2 (v2):

1. **EB over-shrinkage on green-cohort players** — r drops to 0.049 for players with 1–10 prior games.
2. **Team-form drift not modeled** — BBL Esports and Karmine Corp had strong 2025 priors, all 3 models overrated their 2026 players.
3. **Roster churn not modeled** — AMER cohort r=0.147 because 4/12 AMER teams ran 100% new starting fives vs their 2025 squads.

v3 layers five targeted improvements on top of v2's ensemble. Each is validated empirically against the v2 baseline via the multi-holdout backtest (`scripts/backtest_v3.py`).

---

## What it does

Same input/output contract as v2:
- **Input**: Historical CSVs, current roster (player → team/region/role), pickrate summary.
- **Output**: Per-player `SuggestedVP` price for the target stage, snapped to 0.5 increments in [6, 15].

v3 differs in **how the expected-points estimate is built** and **how that estimate maps to a price**:

| Mechanism | v2 | v3 |
|---|---|---|
| EB shrinkage target | Population mean | Role-mean (D/I/C/S) |
| Min shrinkage weight B | None — collapses to ~0 for low n | Floor B ≥ 0.30 |
| Team strength | Pooled WR (no recency) | Decayed across stages: w(t) = 0.6^Δ |
| Roster churn | None | `team_continuity` = fraction of player's prior games on current team |
| Star cap | None | If `recent_avg / historic_avg < 0.85` and `historic > pop_mean`, price ≤ 11.0 |
| Quantile mapping | Single global curve | Region-calibrated curves (AMER may differ) |

---

## Boundaries & edge cases

- **No pickrate weighting added.** Confirmed dead weight in the 2026 Stage 1 backtest.
- **No brand popularity.** Same — circular with team strength.
- **No variance premium.** F6 in v2 spec showed variance is not persistent.
- **Star cap is one-sided** — it only caps over-valuations. Underrated breakouts (Favian, Saadhak, oonzmlp) are not solved by v3; that requires forward-looking info v3 doesn't have.
- **Region-specific quantile curves require ≥ 30 players per region** in calibration. If fewer, fall back to global curve.
- v3 still uses Pts (not PPM) per v2 F2.
- v3 still filters China per v2 baseline.

---

## Testing & verification

### Key scenarios

- [ ] EB shrinkage with B-floor: player with n=3 games at PPG=12 should price ABOVE role-mean (whereas v2 would collapse to pop-mean ~7.5).
- [ ] Team-form decay: player on team whose 2025-late stages had WR=0.3 (recent slump) but 2024 WR=0.6 should receive a lower team-form multiplier than v2's pooled WR.
- [ ] Team-continuity penalty: player with `team_continuity < 0.3` (just moved teams) should be priced closer to role-mean than to their personal historical mean.
- [ ] Star cap: BBL Esports's Lovers rock case. Historic PPG=10, last-stage PPG=6.5, `recent/historic = 0.65 < 0.85` → final price capped at 11.0 (was 13.0 manual, 15.0 Algo2 in the live eval).
- [ ] Region quantile: AMER players priced from AMER-specific quantile if region has ≥30 players.

### Edge cases

- Rookies (n=0): use role-mean directly with no shrinkage math (`B=1` since no observed data). No star cap.
- Player on a brand-new team (e.g. ENVY 2026, no team history): team-form multiplier defaults to 1.0 (no penalty/bonus).
- Single-stage-only history: team-form decay degenerates to a single weight (1.0) and v3's team factor equals current WR.

### Automation notes

- Unit-test each new component independently (`tests/test_v3_*.py`).
- Integration test via `scripts/backtest_v3.py` against 6 historical holdouts; report Pearson r vs actual PPG per cohort.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-05-14 | EB floor B=0.30 (not 0.20 or 0.40) | 0.30 retains meaningful per-player signal for n=3 (B raw ≈ 0.15 → 0.30 doubles signal) without overriding population for n=1. Tunable. |
| 2026-05-14 | Team-form decay base = 0.6^Δ stages | 0.6 gives a 2025 Kickoff weight of ~0.13 by 2026 Stage 1 (4 stages back), with Santiago at ~0.36 and Kickoff 2026 at 0.6. Reasonable half-life. |
| 2026-05-14 | Star cap threshold 0.85, max price 11.0 | Catches JohnQT (0.78), skuba (0.82), Lovers rock (0.65) from the 2026 Stage 1 overrate list. 11.0 is the median elite tier. |
| 2026-05-14 | Continuity threshold 0.3 | Below 0.3 = player has played <30% of prior games on current team → most context is from elsewhere. |
| 2026-05-14 | Drop role-mean shrinkage if `<5 players per role` in training | Avoid degenerate role priors for tiny samples. |

## Related

- v2 spec: `specs/expected-points-model.md`
- 2026 Stage 1 eval: `output/Stage1_2026_pricing_learnings.md`
- v2 code: `v2/expected_points.py`, `v2/pricing.py`
