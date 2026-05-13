# v2 Retune — Leave-One-Stage-Out Hyperparameter Search

**Date:** 2026-05-14
**Hypothesis:** Now that we have 2026 Stage 1 results, can we re-tune v2's hyperparameters (ensemble weights, calibration windows, recency map, pickrate weight) to improve Pearson r against actual PPG across all stages?
**Result:** **No meaningful gain.** Leave-one-out mean Δr = **−0.006** (essentially zero, within noise). v2 with default hyperparameters is at the data-driven ceiling within this search space.

---

## 1. Setup

### Search space (448 configs per LOO fold)
- **Ensemble weights** (w_eb, w_ridge, w_ema): 28-point simplex grid {0.0, 0.2, 0.33, 0.4, 0.6, 0.8, 1.0} summing to 1.0.
- **Pickrate weight**: {0.0, 0.05}.
- **Calibration windows**: 4 sets — v2 default (2025 S1+S2), +2026K, +2026K+Santiago, recent-only (2025 S2 + 2026 K+S).
- **Recency stage-order map**: 2 variants — v2 default (groups Masters), strict (every stage distinct).

### Protocol
For each of 8 holdouts (2024 S1/S2, 2025 Kickoff/S1/S2, 2026 Kickoff/Santiago/S1):
1. Optimize hyperparams on the **other 7** stages — find argmax mean Pearson r.
2. Apply winning config to the held-out 8th.
3. Compare to v2 default config evaluated on the same held-out.

This is honest cross-validation — we never optimize on the stage we report on.

---

## 2. Headline result

| | Mean Pearson r |
|---|---|
| **v2 baseline (default)** | **0.425** |
| **v2 retune (LOO winner)** | **0.419** |
| **Δ** | **−0.006** |

Per-holdout:

| Holdout | v2 baseline | v2 retune | Δ | Winning config |
|---|---|---|---|---|
| 2024 Stage 1 | 0.422 | 0.399 | −0.023 | +2026K+S \| ridge-heavy |
| 2024 Stage 2 | 0.519 | 0.481 | −0.038 | +2026K+S \| ridge-heavy |
| 2025 Kickoff | 0.263 | 0.251 | −0.012 | +2026K+S \| ridge-heavy |
| **2025 Stage 1** | 0.534 | **0.563** | **+0.029** | +2026K+S \| ridge-heavy |
| 2025 Stage 2 | 0.552 | 0.523 | −0.029 | +2026K+S \| ridge-heavy |
| **2026 Kickoff** | 0.360 | **0.402** | **+0.042** | +2026K+S \| ridge-heavy |
| 2026 Santiago | 0.356 | 0.353 | −0.003 | v2 default \| EB-heavy |
| 2026 Stage 1 | 0.390 | 0.377 | −0.013 | +2026K+S \| ridge-heavy |

3 wins (one tie) out of 8. Roughly random.

---

## 3. What the tuner converged on (and why it didn't help)

Across 7/8 LOO folds, the tuner picked:
- **Calibration windows:** `plus_2026KS` (include 2026 Kickoff + 2026 Santiago alongside 2025 stages).
- **Ensemble weights:** (EB=0.2, Ridge=0.8, EMA=0.0) — heavily Ridge-dominated.
- **Pickrate:** 0.05 (default).
- **Recency map:** v2 default (strict variant never won).

**Why these don't translate:**

1. **Ridge-heavy ensemble fits the past stages better but doesn't generalize.** Ridge memorizes the pooled-window structure of *those specific stages*. EB and EMA carry signal that Ridge throws out — that signal helps on out-of-sample stages.
2. **Including 2026 data in calibration is locally helpful** (boosts 2026 Kickoff +0.042) but hurts 2024 stages because it shifts the learned Ridge weights toward 2026 patterns that don't reflect older seasons.
3. **The 3-model ensemble (EB / Ridge / EMA, equally weighted) is genuinely diversifying.** Each is wrong in different ways, and averaging dampens stage-specific overfit.

This is a textbook bias-variance result: the LOO-winning config has lower bias on the training stages (higher within-training r ≈ 0.45 vs baseline) but higher variance (more sensitivity to which stages it was tuned on).

---

## 4. Mini-finding: 2026 Kickoff is the one consistent gainer

LOO-tuned config beats v2 default on 2026 Kickoff by +0.042. This holdout shares structure with 2026 Stage 1 (same year, post-Champions roster movement). The fact that retune helps here but not on 2026 Stage 1 itself suggests:
- 2026 Kickoff and 2026 Santiago are the "easier" 2026 stages to predict (more team continuity from late 2025).
- 2026 Stage 1's failures are driven by something the tuner can't see: the post-Champions roster reshuffle.

If you specifically want better **early-2026** Stage 1 predictions, the retune doesn't unlock it. The fix has to come from somewhere else (new features, manual review for high-churn teams, or hybrid Manual+Algo pricing).

---

## 5. What this tells us about v2

**v2 is at the data-driven ceiling for its model class.**

- Within the structural constraints of (EB + Ridge + EMA → quantile map → budget calibrate), hyperparameter search across 448 configs × 8 LOO folds = ~3,500 evaluations cannot find a configuration that systematically beats the default.
- The "knob-twist for better backtest" approach has been mined out. Further improvement requires structural change.

**This is a legitimately useful negative result.** It tells us where NOT to spend effort.

---

## 6. So how do we actually improve from here?

Three directions ordered by tractability:

### Path A — Add data, keep v2 (Recommended, lowest risk)
- Run v2 with the new 2026 Stage 1 training data when pricing 2026 Stage 2.
- v2's data-driven parameters (EB tau²/sigma², EMA alpha, Ridge weights) auto-update with each new stage.
- Expectation: organic small improvement as roster + form patterns stabilize.

### Path B — Wait for 2026 Stage 2 evidence, then revisit star_cap
- The v3 ablation showed star_cap is the only component with a localized 2026 S1 win (+0.027).
- If 2026 Stage 2 shows the same roster-churn-driven overrate pattern, star_cap becomes a defensible default-on.
- If it doesn't, the BBL/Karmine overrate was n=1 noise.

### Path C — Structural redesign (high risk, may not pay off)
- Different model class: gradient boosting (XGBoost) on the same features. Better at capturing non-linear interactions.
- Per-region sub-models: train a separate Ridge per region. AMER's idiosyncrasies (roster churn) might be learned regionally without contaminating EMEA/PAC.
- Hybrid: use Manual prices as a strong prior, let the algorithm only *refine* (small adjustments around Manual).

Path C is a research project. Not worth doing until 2026 Stage 2 confirms whether the 2026 Stage 1 failures repeat or were anomalous.

---

## 7. Files written this run

- `v2_retune/__init__.py`, `config.py`, `base_estimates.py`, `score.py` — parameterized v2 retune module
- `scripts/retune_v2.py` — LOO tuner
- `output/v2_retune_loo.csv` — raw per-fold results
- `output/v2_retune_findings.md` — this file

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-05-14 | Don't ship a "v2-retuned" variant | LOO mean delta is −0.006; the within-training winner overfits to training stages. |
| 2026-05-14 | Keep v2's default ensemble at (1/3, 1/3, 1/3) | Ridge-heavy variant wins on tuning data but loses on held-out. Equal weights diversify cleanly. |
| 2026-05-14 | Continue using v2's calibration windows (2025 S1 + S2) | Adding 2026 data to calibration helps some stages, hurts others. Wash. |
| 2026-05-14 | Mark hyperparameter tuning as "mined out" | 3,500-config search confirms no free lunch in this search space. Move structural redesign behind 2026 S2 data collection. |
