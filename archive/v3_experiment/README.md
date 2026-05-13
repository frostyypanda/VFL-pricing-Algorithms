# v3 Pricing Experiment (archived 2026-05-14)

Status: **deprecated**. v3 layered 5 structural changes on top of v2 (role-mean EB, B-floor, team-form decay, continuity penalty, star cap, region quantile). 8-holdout backtest + per-component ablation showed every change individually hurt v2; v3-full was −0.135 mean Δr vs v2. See `results/v3_findings.md`.

Kept for reference only. Don't re-enable without new evidence (e.g., 2026 Stage 2 repeating the BBL/Karmine roster-churn pattern).

Structure:
- `v3/` — module source (eb_role, team_form, continuity, star_cap, expected_points, pricing, flags)
- `tests/` — unit tests (19 tests, all passed at time of archive)
- `scripts/backtest_v3.py`, `scripts/ablate_v3.py` — backtest harness + ablation
- `v3-pricing.md` — design spec
- `results/` — backtest CSVs and findings doc
