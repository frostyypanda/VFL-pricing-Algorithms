# v3 Pricing — Multi-Holdout Backtest & Component Ablation

**Date:** 2026-05-14
**Hypothesis:** The 5 improvements proposed in `Stage1_2026_pricing_learnings.md §5` should improve Algo2 (v2) across multiple historical stages, not just on 2026 Stage 1.

**Result:** **All 5 components individually hurt v2 on the majority of holdouts.** v3 (all 5 combined) is worse than v2 on every holdout. Conclusion: the proposed fixes target failure modes that are stage-specific (mainly 2026 Stage 1), and applying them globally amplifies noise more than it removes bias.

---

## 1. Corrections to previous learnings doc

Before running the backtest I verified the data and discovered the previous claim was wrong:

> "AMER started this season's competitive calendar at Stage 1 (no Kickoff/Santiago equivalent), so prior data is thinner..."

**Wrong.** AMER played in 2026 Kickoff (239 played rows, vs EMEA 279 / PAC 273) and 2026 Santiago (45 rows). Pre-Stage-1 game count for AMER players is mean 24.2 (EMEA 24.9, PAC 27.4). **Data depth is comparable across regions.**

The actual driver of AMER's low r is **roster churn**: 4 of 12 AMER teams (ENVY, Evil Geniuses, FURIA, G2 Esports) had 100% new starting fives vs their 2025 squads. The historical data for those players exists; it's just on different teams now. `output/Stage1_2026_pricing_learnings.md` has been updated.

---

## 2. v3 design summary

Five components, each in its own module (`v3/eb_role.py`, `team_form.py`, `continuity.py`, `star_cap.py`, `pricing.py`), each toggled by `V3Flags`:

| Flag | Component | Spec ref |
|---|---|---|
| `role_mean_eb` | EB shrinks toward role-mean (D/I/C/S) instead of population mean | 5.1, 5.3 |
| `b_floor` | EB shrinkage weight B ≥ 0.30 (instead of collapsing to ~0 for n<5) | 5.1 |
| `team_form_decay` | Player base × (decayed_WR / pooled_WR), clamped to [0.7, 1.3] | 5.2 |
| `continuity` | If player's prior games on current team < 70%, blend toward role-mean | 5.5 (new) |
| `star_cap` | If recent_avg / historic_avg < 0.85 AND historic > pop_mean, cap price at 11 | 5.4 |
| `region_quantile` | Quantile mapping per-region (if ≥30 players in region) | 5.5 |

Unit tests for each component all pass (`tests/test_v3_*.py`, 19 tests). The issue isn't bugs — it's that the design assumptions don't hold up empirically.

---

## 3. Multi-holdout backtest (full v3 vs v2 vs others)

Trained-before-target setup: for each holdout stage, train = all played rows strictly before; evaluate predicted price vs actual PPG in the held-out stage.

| Holdout | n | Manual r | Algo1 r | **Algo2 (v2) r** | **v3 r** | Δ (v3 - v2) |
|---|---|---|---|---|---|---|
| 2024 Stage 1 | 168 | n/a | 0.366 | **0.428** | 0.401 | -0.027 |
| 2024 Stage 2 | 171 | n/a | 0.405 | **0.529** | 0.223 | **-0.306** |
| 2025 Kickoff | 182 | 0.536 | 0.379 | 0.249 | **-0.109** | **-0.358** |
| 2025 Stage 1 | 188 | 0.424 | 0.449 | **0.534** | 0.352 | -0.182 |
| 2025 Stage 2 | 184 | 0.505 | 0.458 | **0.562** | 0.388 | -0.174 |
| 2026 Kickoff | 188 | 0.650 | 0.313 | 0.356 | 0.194 | -0.162 |
| 2026 Santiago | 50 | 0.254 | **0.647** | 0.320 | 0.270 | -0.050 |
| 2026 Stage 1 | 168 | **0.572** | 0.343 | 0.383 | 0.235 | -0.148 |

**Average Δ (v3 − v2): -0.176.** v3 is dramatically worse.

(Manual proxy = `Game Start VP` of first played game per player for past stages; actual `Stage1_Price` for 2026 Stage 1. Manual unavailable for 2024 since 2024 CSVs lack VP columns.)

---

## 4. Ablation: which component is the culprit?

Each component turned on alone, on top of v2 baseline. Pearson r vs actual PPG:

| Holdout | v2 base | role_eb | b_floor | team_form | continuity | star_cap | region_q | full v3 |
|---|---|---|---|---|---|---|---|---|
| 2024 S1 | **0.428** | 0.412 | 0.412 | 0.400 | 0.411 | 0.412 | 0.386 | 0.401 |
| 2024 S2 | **0.529** | 0.521 | 0.516 | **0.268** | 0.523 | 0.499 | 0.453 | **0.223** |
| 2025 KO | **0.249** | 0.189 | 0.188 | **-0.090** | 0.135 | 0.164 | 0.181 | **-0.109** |
| 2025 S1 | 0.534 | **0.546** | 0.540 | 0.385 | 0.538 | 0.544 | 0.511 | 0.352 |
| 2025 S2 | **0.562** | 0.560 | 0.555 | 0.467 | 0.542 | 0.522 | 0.486 | 0.388 |
| 2026 KO | **0.356** | 0.335 | 0.335 | 0.231 | 0.322 | 0.324 | 0.342 | 0.194 |
| 2026 SA | **0.320** | 0.279 | 0.312 | 0.291 | 0.226 | 0.312 | 0.312 | 0.270 |
| 2026 S1 | 0.383 | 0.377 | 0.387 | 0.249 | 0.375 | **0.410** | 0.362 | 0.235 |
| **Mean Δr vs v2** | — | -0.011 | -0.013 | **-0.143** | -0.029 | -0.022 | -0.043 | -0.135 |

### Per-component verdict

- **Team-form decay (5.2): rejected.** Costs ~0.14 r on average and crashes 2025 Kickoff to *negative* correlation. The ratio (decayed_WR / pooled_WR) is too volatile across stages — a single below-average stage swings team multipliers by 30%, scrambling the entire player ordering. The "BBL drift" failure that motivated this was real but localized to 2 teams; the multiplier penalizes all 36 teams.
- **Region-specific quantile (5.5b): rejected.** -0.043 mean. The global quantile curve was already well-calibrated; per-region curves overfit to within-region noise.
- **Continuity penalty (5.5a): rejected.** -0.029 mean. Pulling new-team players toward role-mean discards their personal signal; their prior PPM on the old team is still informative.
- **Star cap (5.4): conditional — keep optional.** -0.022 mean but **+0.027 on 2026 Stage 1** (the only stage with the BBL/Karmine overrate pattern). Win-rate 1/8.
- **Role-mean EB (5.1a): near-neutral.** -0.011 mean, slight win on 2025 S1 (+0.012). Plausible to keep but no clear benefit.
- **B-floor (5.1b): near-neutral.** -0.013 mean. Slight wins on 2026 Stage 1 (+0.004) and 2025 S1 (+0.006). Doesn't hurt much but doesn't help.

---

## 5. Why the changes hurt — root cause analysis

The 2026 Stage 1 failure modes are real but **rare**. The proposed fixes were extrapolated from a single stage:

1. **Team-form drift** was observed in BBL and Karmine Corp (2 teams). The fix penalizes all 36 teams whenever their recent WR differs from their pooled WR — which happens routinely for non-drift reasons (stage variance, opponent variance, who they played early vs late).

2. **AMER cohort failure** was attributed to "data thinness" (wrong) and "roster churn" (partial). The continuity fix penalizes players who moved teams by pulling them toward role-mean — but their old-team PPM is still the best estimate of their personal skill. The team_factor handles the team context separately.

3. **Green-cohort over-shrinkage** is real but the B-floor's fix (force 30% personal signal) introduces noise that hurts more than it helps because for n<5, the personal signal is dominated by sampling variance.

The underlying pattern: **the 2026 Stage 1 backtest was n=1 evidence**. Generalizing 5 corrections from one observation is overfitting to a sample of one.

---

## 6. What should we actually do?

### Recommendation: keep v2 as baseline. Treat v3 as instructive failure.

1. **Don't ship v3.** It's clearly worse on average.
2. **Star cap is the one defensible win on 2026 Stage 1.** If we believe the post-Champions VCT environment has more roster churn than past seasons (an empirical question — needs 2026 Stage 2 data to confirm), star cap could be useful. For now: keep the code, don't enable by default.
3. **The right fix for team-form drift is probably not a multiplier.** It's a per-team intervention that should fire only when roster turnover or known team-context changes are detected (e.g., a starter swap that affected only one player's expected role).
4. **For "team that just changed rosters"** (ENVY-class, 100% new lineup), the model should fall back to role-mean priors *automatically* when n_team_continuity = 0, but should not penalize individual movers (continuity 0 < x < 1).
5. **Re-run this analysis after 2026 Stage 2.** With one more data point we can distinguish "2026 Stage 1 was anomalous" from "post-Champions seasons systematically have more roster churn."

### One discipline lesson

The original learnings doc proposed 5 changes from 1 backtest. The right discipline is: **propose ≤1 change per N=1 observation**, ablate, and only ship after multi-holdout confirmation. v3 should have been built one flag at a time, not all five at once.

---

## 7. Files written this run

- `v3/__init__.py`, `eb_role.py`, `team_form.py`, `continuity.py`, `star_cap.py`, `expected_points.py`, `pricing.py`, `flags.py`
- `tests/test_v3_eb_role.py`, `test_v3_team_form.py`, `test_v3_continuity.py`, `test_v3_star_cap.py` — all 19 tests pass
- `scripts/backtest_v3.py` — multi-holdout backtest (Manual/Algo1/Algo2/v3)
- `scripts/ablate_v3.py` — per-component ablation
- `output/v3_backtest_summary.csv` — backtest raw results
- `output/v3_ablation_summary.csv`, `v3_ablation_pearson_pivot.csv` — ablation raw
- `specs/v3-pricing.md` — design spec (status updated to **deprecated** below)

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-05-14 | Fix incorrect AMER data-thinness claim in `Stage1_2026_pricing_learnings.md` | AMER pre-S1 data is comparable to other regions (239 Kickoff rows). Real driver is roster churn. |
| 2026-05-14 | Ship v3 as toggleable (V3Flags) rather than a fixed pipeline | Allows ablation; v2 stays as baseline. |
| 2026-05-14 | Don't enable v3 by default; mark spec as deprecated | Multi-holdout backtest shows v3 strictly worse than v2 on average. |
| 2026-05-14 | Keep `star_cap` available but not enabled | Conditional win on 2026 Stage 1; needs more data to justify default-on. |
| 2026-05-14 | Drop team_form_decay entirely | Costs ~0.14 r and tanks 2025 Kickoff to negative correlation. Wrong mechanism. |
