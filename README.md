# VFL Pricing Algorithms

Pricing algorithms and tooling for **Valorant Fantasy League (VFL)** — a community fantasy game built on top of the Valorant Champions Tour (VCT). Managers draft 11-player squads under a 100 VP budget and score points from real VCT match performances.

- VFL site: https://www.valorantfantasyleague.net
- Stats source: https://www.vlr.gg

This repo is an attempt to build *better* player price lists than the current VFL pricing — measured by spread health, archetype diversity, and predictive accuracy against actual VCT performance.

---

## What's in here

```
.
├── data/        Per-season CSVs (2024, 2025, 2026) + pickrate / IGL data
├── v2/          Current pricing engine: data loaders, scoring, EP model, ILP optimizer, scraper
├── scripts/     One-off per-gameweek runners (W1, GW3..GW5 reports + scrapes)
├── archive/     Older algo versions and analysis (kept for reference)
├── output/      Generated pricing sheets, transfer plans, and reports
├── specs/       Spec-anchored feature/decision docs
├── tests/       Test suite for the v2 engine
└── CLAUDE.md    Full project context (game rules, scoring, algorithms, conventions)
```

---

## Game rules at a glance

- **Squad:** 11 players (2 Duelist / 2 Controller / 2 Initiator / 2 Sentinel / 3 Wildcard), max 2 per VCT team
- **Budget:** 100 VP (player prices 6–15 VP)
- **IGL:** designate 1 captain per gameweek — their points are doubled
- **Regions:** Americas, EMEA, Pacific (China is excluded from VFL)
- **Format we focus on:** regular Stage 1 / Stage 2 round-robin play

Scoring blends per-map kill thresholds, multi-kill round bonuses, map win/loss points, and VLR rating bonuses. Full table in [`CLAUDE.md`](./CLAUDE.md).

---

## Pricing approach

Six algorithms live in `archive/pricing_algorithms.py` (legacy) and the production engine in `v2/`:

1. **Baseline EMA** — exponential moving average of points-per-map, linear map to [6, 15]
2. **Distribution-aware** — same scores, but force a healthy quantile spread (fixes median-below-9)
3. **Role-adjusted** — proxy roles from scoring patterns; discount duelists, boost supports
4. **Team strength** — split expected points into personal vs team-based, weight by team win rate
5. **Variance-aware** — premium for consistent scorers, discount for boom-or-bust
6. **Combined** *(recommended)* — weighted blend of all signals (60% EMA + 12% team + 8% pickrate + 8% brand + 7% opponent + 5% consistency) with quantile mapping

Generated prices are evaluated on price accuracy (MAE / RMSE / correlation), composition diversity (greedy 11-player optimizer with random restarts), and a backtest against actual Stage 1 results.

---

## Running the per-gameweek tooling

The `scripts/` directory contains one-off runners used during the live 2026 season. Run them from the repo root so the `data/` and `output/` folders resolve correctly:

```bash
# Scrape a gameweek's VLR results
python scripts/scrape_gw4.py

# Generate the next-gameweek transfer plan
python scripts/generate_gw5_report.py
```

Each script writes to `data/w<N>_vlr_results.json` (scrapers) or `output/GW<N>_transfer_plan.md` (planners).

---

## More detail

- Full game rules, scoring tables, data schema, and algorithm internals: [`CLAUDE.md`](./CLAUDE.md)
- Per-feature specs and cross-cutting decisions: [`specs/`](./specs)
