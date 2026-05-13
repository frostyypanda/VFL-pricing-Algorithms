# v2 Hyperparameter Retune Experiment (archived 2026-05-14)

Status: **mined out**. Leave-one-stage-out search over ensemble weights, calibration windows, recency stage-order map, and pickrate weight (~3,500 evaluations) produced LOO mean Δr = −0.006 vs v2 baseline — within noise. v2's defaults (equal-weight EB/Ridge/EMA ensemble, 2025 S1+S2 calibration) are at the data-driven ceiling for this model class. See `results/v2_retune_findings.md`.

Kept for reference. Don't redo unless the model class changes (e.g., XGBoost replacement) or major new training data arrives.

Structure:
- `v2_retune/` — parameterized recalibration module (config, base_estimates, score)
- `scripts/retune_v2.py` — LOO tuner
- `results/` — per-fold CSV + findings doc
