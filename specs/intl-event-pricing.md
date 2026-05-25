# Feature: International Event Pricing (Masters / Champions)

## Status
draft

## Why this exists

The regular VFL pricing pipeline (specs/pricing-algorithm.md) is built for the regional Stage 1 / Stage 2 format: 11-player squads, 100 VP budget, 6-15 VP range, 6 regional gameweeks. International events (Masters, Champions) use a different game design:

- **Smaller squad** (6 players instead of 11)
- **Smaller budget** (50 VP instead of 100)
- **Compressed VP range** (5-13 instead of 6-15)
- **Cross-regional pool** including China teams
- **No regional schedule** — every player plays every gameweek

The regional-pricing engine cannot be used directly because it (a) hard-codes squad size/budget, (b) excludes China from the population, and (c) accounts for AMER GW1 byes. This spec defines the international-event extension.

---

## What it does

Takes the 12 qualified Masters London teams (3 per region × AMER/EMEA/PAC/CN, 5 players each → ~60 players) and produces a VP price list constrained to:

- Budget: **50 VP** total
- Squad: **6 players** (1D + 1C + 1I + 1S + 2 Wildcards)
- VP range: **[5.0, 13.0]** in 0.5 increments
- Target mean: **8.33 VP** (= 50/6)

Each row in the output CSV includes a per-player **uncertainty score (0-1)** and **flag (Y/N)** so managers can identify volatile picks.

---

## Architecture

### Step 1: Roster construction

Maintain a `QUALIFIED_TEAMS` dict mapping team → list of `(player, role)` for the 12 teams that qualified. Roster sourced from vlr.gg/Liquipedia event page. Player names follow the historical DB names where possible; aliases (e.g., `spikeziN` → `Spike`) are stored in `PLAYER_NAME_ALIASES`.

### Step 2: Base value estimation

For each player, compute three estimators and take the unweighted mean:

1. **Empirical-Bayes shrunk avg Pts** — uses population mu/tau²/sigma² estimated from all 2024-2026 played data.
2. **Recency-weighted avg Pts** — weights by (stage × year): newer years and later stages (Stage 1, Masters) count more.
3. **Stage 1 + Santiago mean × 1.15** — applied only if the player has ≥1 recent game; small extra premium for current form.

For players with 0 historical games, fall back to a role-based prior (`{D: 9.0, I: 7.0, C: 7.0, S: 7.5}`).

### Step 3: Regional adjustment

A multiplicative factor per region, derived from `output/regional_scaling.json` (output of `scripts/analyze_regional_scaling.py`). The factor is the median ratio of (international Pts/game) / (regional Pts/game) across players who played both.

**Important:** the ratio is biased upward by selection (only top regional players reach internationals). The CN factor is further reduced by 5% to account for this selection bias, since the priced pool includes mid/bottom CN starters.

### Step 4: Uncertainty score

A 0-1 score driven by:

| Component | Contribution |
|-----------|--------------|
| `n_total < 5` | +0.50 |
| `5 ≤ n_total < 12` | +0.25 |
| `n_recent < 3` (2026 S1+Santiago) | +0.25 |
| Region=CN, n_intl < 5 | +0.30 |
| Region=CN, n_intl ≥ 5 | +0.15 |
| Region≠CN, n_intl < 3 | +0.20 |

Flag = Y when score ≥ 0.30. The score is monotonic and explainable via the `UncertaintyReason` column.

### Step 5: Quantile mapping → VP

Rank all 60 players by adjusted value (percentile rank, first-method tie-break). Interpolate against a calibrated quantile curve targeting:

- 10% of players at 5.0-6.0 (cheap fillers)
- ~40% at 7.0-9.0 (the value zone)
- 8-12% at 11.0+ (premium stars)

Snap to 0.5 increments within [5.0, 13.0]. Iteratively shift toward `mean = 8.33` until within 0.05.

---

## Boundaries & edge cases

### What this does NOT do
- Does not optimize a squad (the existing team optimizer is for the 100 VP / 11 player format and would need its own extension).
- Does not update prices dynamically between gameweeks of the same event.
- Does not handle stand-ins or roster changes during the event.

### Edge cases
- **New player (0 games)**: Falls back to role-based prior, gets ~5-6 VP and Flag=Y.
- **Player with no team history but DB history (e.g., transfer)**: `find_player_history` falls back to any-team lookup. Price reflects past performance regardless of current team.
- **CN player priced higher than EMEA player**: Possible — the CN factor is mild (~5%). If domain expertise suggests a stronger CN discount, raise the post-selection adjustment in `load_regional_factor()`.

---

## Testing & verification

### Key scenarios
- [ ] Generated mean within 0.05 of 8.33
- [ ] Full range [5.0, 13.0] used
- [ ] All 60 players have a non-null price
- [ ] No bucket > 20% of players (avoids price clustering)
- [ ] All Flag=Y players have a non-empty `UncertaintyReason`
- [ ] CN players average below AMER players (sanity: ~5-10% lower on average)

### Backtest
- Run the pipeline pre-Santiago (treat Santiago as held-out) and compare predicted prices to actual Santiago-event manual prices in `archive/santiago_prices_2026.csv`. Target: r > 0.55, MAE < 1.5.

---

## Decision log

| Date | Decision | Why |
|------|----------|-----|
| 2026-05-25 | Use 5-13 VP range (not 5-15) | User requirement matches Masters London VFL config |
| 2026-05-25 | 6-player squad with 1-of-each role + 2 wildcards | User-confirmed format |
| 2026-05-25 | CN factor = median × 0.95 (small extra discount) | Selection bias in raw ratio; priced pool includes weaker CN starters |
| 2026-05-25 | Uncertainty flag threshold = 0.30 | Catches new-team-Chinese players, rookies, players with <5 games |
| 2026-05-25 | EB + weighted + recent ensemble (no Ridge) | Sample is only 60 players; Ridge needs more |

---

## Related
- `specs/pricing-algorithm.md` — the regional pricing pipeline that this extends
- `scripts/price_masters_london.py` — implementation
- `scripts/analyze_regional_scaling.py` — produces the regional factor input
- `output/regional_scaling.json` — current factors
- `output/Masters_London_2026_prices.csv` — generated prices
