# Feature: Expected Points Model

## Status
draft

## Why this exists

The expected points model is the foundation of everything — pricing, team optimization, and evaluation all depend on accurate per-player, per-gameweek point estimates. A bad expected points model makes everything downstream wrong.

The previous implementation used hand-tuned parameters (EMA alpha=0.3, year weights 0.5/1.0/2.0, shrinkage constant k=4, opponent adjustment ±15%). None of these were justified by data. This spec defines an approach where **every parameter is learned from the historical data**.

---

## What it does

Given historical VFL game data, produce for every player a per-gameweek expected fantasy points estimate.

**Input**: Historical CSVs (2024, 2025, 2026), current roster with team/region/role, GW schedule with matchups.

**Output**: A matrix of `expected_pts[player][gw]` for all ~180 players × 6 gameweeks, plus uncertainty estimates (low/high confidence per player).

---

## Empirical Findings That Drive Design

These are from rigorous analysis of the actual VFL data. Every design choice below references a finding.

### F1: Stage-to-stage autocorrelation is moderate
Median r = 0.38 across all consecutive stage pairs. Performance is somewhat predictable but very noisy. Implication: **don't overfit to recent form — there's a ceiling on predictability.**

### F2: Pts is a better predictor than PPM
Kickoff→Stage 1: Pts r=0.54, PPM r=0.44. Implication: **use raw Pts (total fantasy points per game), not PPM (points per map).**

### F3: Optimal recency weights depend on available data
- With only Kickoff data: ~70% old / 30% new is best
- With a full stage of recent data: ~0-15% old / 85-100% new is best
- Implication: **recency weighting must be adaptive, not fixed.** Use regularized regression to learn this.

### F4: Prediction quality jumps at 6+ games
r improves from 0.39 (1-5 games) to 0.50 (6-10 games). Beyond 10 games, marginal returns. Implication: **heavy shrinkage below 6 games, moderate 6-10, light above 10.**

### F5: Team-based points (T.Pts) are 32% of scoring but nearly unpredictable
Stage-to-stage T.Pts correlation is r=0.13. Win rate predicting future T.Pts: r=0.17. Implication: **team strength adjustments should be small.** Don't add a big team multiplier — it adds noise, not signal.

### F6: Variance is not persistent
Player std dev across stages: r=0.08. Implication: **"popoff potential" or ceiling-based pricing adjustments have no empirical basis.** A high-variance player this stage is no more likely to be high-variance next stage.

### F7: Opponent effect is real but moderate
Winners score +5.3 pts vs losers. Being on a strong team is worth ~3 pts/game (split ~equally personal and team). Implication: **opponent adjustment matters but the effect is ~20-30% of total variance, not more.**

### F8: Scoring environment shifted between years
2024 mean Pts = 6.64, 2025 = 7.67, 2026 = 7.82. Implication: **raw 2024 averages need adjustment before blending with 2025/2026.** But normalizing doesn't improve prediction — the regression approach handles this implicitly.

---

## Architecture

### Step 1: Feature Construction

For each player, compute features from all available historical data (games where P?=1, China teams filtered out):

| Feature | Description | Source |
|---------|-------------|--------|
| `avg_pts_recent` | Mean Pts from most recent stage/event | Pts column |
| `avg_pts_prev_stage` | Mean Pts from the stage before that | Pts column |
| `avg_pts_prev_year` | Mean Pts from all games in the prior year | Pts column |
| `n_games_recent` | Number of games in most recent stage | Count |
| `n_games_total` | Total historical games | Count |
| `team_win_rate` | Current team's recency-weighted win rate | W/L column |
| `role` | D / I / C / S (one-hot encoded) | From roster data |
| `avg_pts_team` | Team-average Pts across all their players | Pts column |

No pickrate features in the expected points model — pickrate reflects demand, not expected performance (r=0.11 per finding). Pickrate belongs in the pricing step, not here.

No variance/ceiling features — variance is not persistent (F6).

### Step 2: Empirical Bayes Shrinkage

Before feeding into regression, apply normal-normal empirical Bayes to each player's observed mean Pts. This handles the small-sample problem (F4) without hand-tuning.

**Population parameters (estimated from data):**
```
mu = mean of all player averages (population mean)
tau² = variance of player averages (between-player variance, "true skill spread")
sigma² = pooled within-player variance (game-to-game noise)
```

**Per-player shrunk estimate:**
```
B_i = tau² / (tau² + sigma² / n_i)
shrunk_pts_i = B_i * observed_mean_i + (1 - B_i) * mu_group
```

Where `mu_group` is the role-specific mean (hierarchical: player → role → population).

Key: **tau² and sigma² are computed from the data, not hand-set.** The shrinkage strength is automatically calibrated.

For new players (n_i = 0): `shrunk_pts_i = mu_group` (pure prior).

For new players with tier-2 vlr.gg stats: convert their KPR to approximate Pts using the observed KPR-to-Pts relationship in tier-1 data, apply a tier-2 deflation factor (~0.75-0.85, calibrated from historical promotees if available), then treat as if they have `n_effective = 2-3` games.

### Step 3: Regularized Regression (RidgeCV)

Instead of hand-picking weights for features, let ridge regression learn them from walk-forward cross-validation.

**Training procedure:**
1. For each backtestable window (e.g., predict 2025 Stage 1 from all prior data):
   - Construct feature matrix X from prior data
   - Target y = actual average Pts in the target stage
   - Fit `RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=None)` — `cv=None` means efficient leave-one-out
2. The learned coefficients are the data-driven feature weights
3. The selected alpha is the data-driven regularization strength

**Walk-forward windows available:**
- Window 1: Train on 2024 all → Predict 2025 Stage 1
- Window 2: Train on 2024 + 2025 pre-Stage 2 → Predict 2025 Stage 2
- Window 3: Train on 2024 + 2025 all + 2026 pre-Stage 1 → Predict 2026 Stage 1

**Evaluation**: Average MAE and correlation across all windows. Use bootstrap CIs to assess significance.

### Step 4: Opponent-Adjusted Per-GW Estimates

The Ridge model gives a **base expected Pts per game** for each player. To get per-GW estimates, adjust for the specific opponent:

```
E[pts_i_gw] = base_pts_i × opponent_factor(opponent_team_gw)
```

Where `opponent_factor` is a simple multiplicative adjustment:
```
opponent_factor = 1.0 + beta × (0.5 - opponent_win_rate)
```

The coefficient `beta` is calibrated: using within-stage game-level data, regress actual Pts on opponent win rate and estimate beta. Expected range: 0.2-0.5 (per F7, the effect is moderate). **Beta is learned from data, not hand-picked.**

For GWs where a player's team doesn't play: `E[pts] = 0`.

### Step 5: Simple Ensemble (insurance)

As a safety net, also compute:
- **Model A**: Pure empirical Bayes shrunk average (no regression features)
- **Model B**: RidgeCV prediction
- **Model C**: Simple recency-weighted average (EMA with grid-searched alpha)

Final estimate = average of all models (equal weight, or inverse-MAE weight from walk-forward CV if one model is clearly better).

Equal-weight averaging is robust and well-supported by research — it protects against any single model being badly miscalibrated.

---

## Boundaries & Edge Cases

### What this does NOT do
- Does not predict roster changes, benchings, or meta shifts
- Does not model map-specific performance (e.g., player is better on Ascent)
- Does not use pickrate (that's a pricing signal, not an expected-points signal)
- Does not use variance/ceiling (not persistent per F6)

### Edge cases
- **Player with 0 games**: Use role-group prior from empirical Bayes. If tier-2 stats available, use those with deflation factor.
- **Player who changed teams**: Use their personal stats (P.Pts-based), but note team change increases uncertainty. If they joined a much stronger/weaker team, adjust team_win_rate feature accordingly.
- **International-only data**: Players who only appear in Masters/Champions events have biased stats (international competition is different). Downweight international-only data slightly (this should fall out of the recency feature construction naturally).
- **2024 data with no VP columns**: Fine — we only use Pts, P.Pts, T.Pts, W/L, which exist in all years.

---

## Testing & Verification

### Key Scenarios
- [ ] Walk-forward backtest Window 1: Predict 2025 S1 from pre-S1 data. Target: correlation > 0.50, MAE < 2.5
- [ ] Walk-forward backtest Window 2: Predict 2025 S2. Target: correlation > 0.50, MAE < 2.5
- [ ] Shrinkage behaves correctly: player with 2 games → estimate near population mean. Player with 30 games → estimate near observed mean.
- [ ] New player (0 games) gets role-appropriate prior, not 0 or garbage
- [ ] Opponent adjustment: player facing weak team gets higher estimate than same player facing strong team
- [ ] GW scheduling: AMER players get 0 for GW1, non-AMER get 0 for GW6
- [ ] No hand-tuned parameters: all alphas, shrinkage constants, weights are learned

### Validation Metrics
- **MAE** of predicted avg Pts vs actual avg Pts (primary)
- **Spearman rank correlation** (are the rankings right?)
- **Calibration by tier**: are top-10 predicted players actually the top-10 scorers? What about top-50?
- **Bootstrap 95% CI** on MAE to assess significance

---

## Decision Log

| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Use Pts not PPM as primary signal | PPM r=0.44 vs Pts r=0.54 in backtests | Data |
| 2026-04-06 | Empirical Bayes over hand-tuned k | tau²/sigma² estimated from data, not guessed | Research |
| 2026-04-06 | RidgeCV over manual feature weights | LOO-CV learns weights; prevents overfitting with ~180 samples | Research |
| 2026-04-06 | No variance/ceiling features | Variance not persistent (r=0.08 between stages) | Data |
| 2026-04-06 | No pickrate in expected points | Pickrate→performance r=0.11, not predictive | Data |
| 2026-04-06 | Small opponent adjustment | Effect is real (~3 pts) but team strength unstable (r=0.13) | Data |
| 2026-04-06 | Simple ensemble of 3 models | Insurance against miscalibration; equal-weight proven robust | Research |

---

## Related
- `specs/team-optimizer.md` — consumes this model's output
- `specs/pricing-algorithm.md` — uses expected points + adjustments to set VP prices
- `empirical_analysis.py` — the data analysis that produced the findings above
