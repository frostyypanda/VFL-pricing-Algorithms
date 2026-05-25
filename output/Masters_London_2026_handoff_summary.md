# Masters London 2026 — Pricing Handoff Summary

Generated 2026-05-25.

## Pull & data state

- Pulled 29 missing commits — local repo now has the v2 engine (`v2/`), specs, scripts, and the standardized 2024-2026 CSVs.
- Stale local `data/`/`scripts/` were backed up to `/Users/kei/VFL-pricing-Algorithms-LOCALBACKUP` before the pull.

## Database updated (`data/2026 VFL.csv`)

- Scraped **36 Stage 1 playoff matches** (12 each for AMER/EMEA/PAC) via `scripts/scrape_stage1_playoffs.py` → `data/stage1_playoffs_results.json` (360 player entries).
- Scraped **48 China Stage 1 matches** (full group + seeding + playoffs) → `data/stage1_china_results.json` (480 player entries).
- Merged into the CSV via `scripts/merge_stage1_playoffs.py`. Updated 333 placeholder rows, inserted 27 new playoff rows, replaced 780 China placeholders with 480 real China Stage 1 records.
- Backup at `data/2026 VFL.csv.bak`.

## Regional scaling analysis

- Script: `scripts/analyze_regional_scaling.py`
- Output: `output/regional_scaling.json`, `output/regional_scaling_perplayer.csv`
- 589 player-years with both regional + international data.
- Median (intl_avg / regional_avg) ratios: **AMER 1.023, EMEA 0.970, PAC 1.005, CN 1.031**.
- Naive ratios don't show a big CN penalty — but the data has strong selection bias (only the top regional players ever reach internationals). For pricing we therefore apply `CN × 0.95` on top of the raw ratio, landing CN's effective factor at **0.979**.

## Masters London 2026 pricing

- Output CSV: `output/Masters_London_2026_prices.csv`
- Summary: `output/Masters_London_2026_pricing_summary.md`
- 12 qualified teams × 5 players = **60 players priced**
  - AMER: G2, LEVIATÁN, NRG
  - EMEA: Heretics, Vitality, FUT
  - PAC: Paper Rex, FULL SENSE, Global Esports
  - CN: EDward Gaming, Xi Lai Gaming, Dragon Ranger Gaming
- Format: 50 VP / 6 players (1D + 1C + 1I + 1S + 2 Wildcards) / 5.0-13.0 VP in 0.5 increments

### Distribution

| Metric | Value |
|--------|-------|
| Mean | **8.333** (target 8.333) |
| Median | 8.5 |
| Min / Max | 5.0 / 13.0 |
| Unique prices | 16 of 17 buckets |

### Region averages

| Region | n | Mean VP | Top 3 |
|--------|---|---------|-------|
| AMER | 15 | 9.40 | 13.0, 11.5, 11.0 |
| EMEA | 15 | 8.40 | 12.0\*, 10.5, 10.5 |
| PAC  | 15 | 8.10 | 12.5, 11.5, 11.0 |
| CN   | 15 | 7.43 | 10.0, 10.0, 8.5 |

CN comes out as the lowest-mean region (as expected given the regional translation concern). AMER comes out highest — reflective of G2/LEVIATÁN/NRG being viewed as the strongest 3-team slate.

### Uncertainty columns

Three columns in the CSV identify volatile picks: `Uncertainty` (0-1 score), `UncertainFlag` (Y/N at score ≥ 0.30), `UncertaintyReason` (text).

5 players flagged Y:

| Player | Team | Region | Role | VP | Reason |
|--------|------|--------|------|-----|--------|
| koshmaras | Team Heretics | EMEA | C | 12.0 | 6 games, 0 intl history — high price + thin sample |
| s0pp | FUT Esports | EMEA | I | 10.5 | 3 games, 0 intl history |
| Sayonara | Team Vitality | EMEA | S | 10.0 | 9 games, 0 intl history |
| Crws | FULL SENSE | PAC | I | 5.0 | 4 games, 0 intl history |
| Jieni7 | EDward Gaming | CN | I | 7.0 | 10 games, 0 intl, CN region |

CN players with established intl history (≥5 intl games) get a milder 0.15 uncertainty (not flagged Y), to keep the flag list actionable.

## Spec

Added `specs/intl-event-pricing.md` — the durable contract for this pipeline per the spec-anchored development rule.

## Open caveat for your review

The Masters London event page on vlr.gg currently shows only 2 teams per region (8 total) but Liquipedia and the regional event pages confirm all 12 teams qualified (3 per region). I priced all 12. If the format ends up being 8-team / 2-per-region, filter to the 2nd/3rd seeds only.

## Files created / modified

```
M  data/2026 VFL.csv                                (merged playoffs + China)
+  data/2026 VFL.csv.bak                            (pre-merge backup)
+  data/stage1_playoffs_results.json
+  data/stage1_china_results.json
+  scripts/scrape_stage1_playoffs.py
+  scripts/merge_stage1_playoffs.py
+  scripts/analyze_regional_scaling.py
+  scripts/price_masters_london.py
+  specs/intl-event-pricing.md
+  output/Masters_London_2026_prices.csv
+  output/Masters_London_2026_pricing_summary.md
+  output/Masters_London_2026_handoff_summary.md     (this file)
+  output/regional_scaling.json
+  output/regional_scaling_perplayer.csv
```
