# Feature: Pricing Algorithm

## Status
draft

## Why this exists

Prices are not predictions — they are game design. The goal is not to perfectly predict future performance (impossible, r≈0.38 between stages). The goal is to produce a price list where:

1. Prices roughly reflect expected value (so informed managers have an edge)
2. The distribution enables multiple viable team archetypes (balanced, top-heavy, stars-and-scrubs)
3. Every player feels worth considering at their price
4. The budget constraint (100 VP / 11 players) creates meaningful tradeoffs

The previous implementations hand-tuned every weight (60% EMA, 12% team, 8% pickrate...). This spec defines a pipeline where the expected-points model provides the data-driven core, and the pricing step is a principled mapping from expected value to VP.

---

## What it does

Takes the expected points matrix from the Expected Points Model and converts it to VP prices in [5.0, 15.0] with 0.5 increments.

**Input**: Expected points per player per GW, pickrate data, roster data (team, region, role).

**Output**: VP price per player (5.0-15.0, 0.5 increments), with mean ≈ 9.09.

---

## Empirical Findings That Drive Design

### F1: Manual prices correlate r=0.76 with actual performance
The human pricing committee is already quite good. Our algorithm should aim to match or exceed this, not reinvent from scratch.

### F2: Balanced archetypes (7-12 VP) consistently win
Top-heavy builds underperform. The price distribution should ensure enough players in the 7-12 range that balanced builds are viable, while still having meaningful differentiation.

### F3: The 7-9 VP tier is the value zone (Pts/VP = 0.90)
This is where market inefficiency lives. The pricing should avoid cramming too many strong players at identical prices in this range — create differentiation so managers have real choices.

### F4: Current VFL distribution is right-skewed, mode at 7.5
This is the status quo. Our distribution should be similar in shape but with better spread in the middle range and the full [5, 15] range used.

### F5: Pickrate barely correlates with performance (r=0.11)
But it does reflect demand. Popular players (high pickrate) create competitive tension — many managers want them, which makes them harder to build around. A small pickrate premium makes the game more interesting, but it should be small since pickrate ≠ skill.

### F6: Duelists average 11.2 VP vs 7.7-8.8 for other roles
This is a real structural feature of VFL scoring: duelists naturally score more kill points. The pricing should reflect this (duelists ARE more expensive), not artificially flatten it.

---

## Architecture

### Step 1: Base Value from Expected Points

Start with each player's **season expected value** = sum of E[pts] across all 6 GWs. This comes from the Expected Points Model spec.

The season value naturally accounts for:
- GW1: AMER players score 0 (reduces their season value)
- GW6: non-AMER players score 0 (reduces their season value)
- Opponent matchups per GW

### Step 2: Pickrate Adjustment (small)

Add a small premium for highly-picked players. This reflects demand, not skill — it makes popular players slightly more expensive, creating a "cost of popularity" that adds strategic depth.

```
adjusted_value_i = season_value_i × (1 + pickrate_weight × normalized_pickrate_i)
```

Where:
- `normalized_pickrate_i` = player's pickrate percentile (0 to 1)
- `pickrate_weight` is **learned**: fit against the relationship between pickrate and actual VFL dynamic prices. Expected range: 0.02-0.08 (small, per F5). If backtesting shows it hurts, set to 0.

For players with no pickrate data: use 0 (no adjustment).

### Step 3: Map to VP via Quantile Scaling

This is the core pricing function. Instead of linear mapping (which compresses the distribution) or VORP (which can be unstable), use **quantile-based mapping** calibrated to produce the target distribution.

**Procedure:**
1. Rank all ~180 players by adjusted_value
2. Assign percentile ranks (0 = worst, 1 = best)
3. Map percentile to VP using a calibrated quantile function

**The quantile function is defined by anchor points:**

| Percentile | VP |
|------------|-----|
| 0.00 | 5.0 |
| 0.05 | 5.5 |
| 0.10 | 6.0 |
| 0.15 | 6.5 |
| 0.25 | 7.5 |
| 0.40 | 8.0 |
| 0.50 | 8.5 |
| 0.60 | 9.0 |
| 0.70 | 9.5 |
| 0.80 | 10.5 |
| 0.85 | 11.0 |
| 0.90 | 12.0 |
| 0.95 | 13.5 |
| 1.00 | 15.0 |

Between anchor points, use linear interpolation. Then snap to nearest 0.5.

**Why this specific function?** It is calibrated to produce:
- Mean ≈ 9.0-9.1 (budget average)
- Median ≈ 8.5 (matches observed VFL distributions)
- Mode in the 7.5-8.5 range (the "value zone")
- ~10% of players at 5.0-6.0 (cheap options)
- ~5-8% of players at 13.0-15.0 (premium stars)
- Smooth coverage of all 0.5 increments (no dead zones)

**However**: these anchor points are an initial estimate. They should be **calibrated** by:
1. Checking the resulting mean — adjust anchors if mean drifts from 9.09
2. Running the team optimizer — verify that balanced, top-heavy, AND stars-and-scrubs archetypes all produce teams within 5% of optimal
3. Comparing against manual prices — correlation should be > 0.70

If the optimizer shows one archetype dominating, adjust the curve to spread the value zone.

### Step 4: Budget Calibration

After snapping to 0.5 increments, verify:
- Mean price is within 0.1 of 9.09
- If not, shift all prices by a constant and re-snap

This is a simple post-hoc adjustment, not a design parameter.

### Step 5: Distribution Verification

Run automated checks:
- [ ] Mean in [8.9, 9.2]
- [ ] Median in [8.0, 9.0]
- [ ] Min = 5.0, Max = 15.0
- [ ] Every 0.5 increment from 5.0 to 15.0 has at least 1 player
- [ ] No single 0.5 bucket has > 20% of all players
- [ ] The top 11 players by price sum to > 120 VP (stars are expensive enough to force tradeoffs)
- [ ] An 11-player team at 100 VP is not trivially easy to build with all top players

---

## Boundaries & Edge Cases

### What this does NOT do
- Does not set prices for international events (different budget, different rules)
- Does not dynamically update prices during a stage (this is pre-stage pricing only)
- Does not try to match the VFL committee's manual prices exactly — the goal is better prices, not identical prices

### Edge cases
- **Player with 0 season value** (e.g., team doesn't play at all): Price at VP_MIN (5.0)
- **All AMER players have lower season value** because they miss GW1: The quantile mapping handles this naturally — they compete within their percentile rank, which already accounts for their 0-GW1
- **Very few players at a price point**: Acceptable if the point is at the extremes (5.0 or 15.0). Mid-range should be populated.

---

## Testing & Verification

### Key Scenarios
- [ ] Generated prices mean is within 0.1 of 9.09
- [ ] Full range [5.0, 15.0] is used
- [ ] Correlation with manual 2026 S1 prices > 0.65
- [ ] Team optimizer produces viable teams under generated prices (budget ≤ 100, all constraints met)
- [ ] Multiple archetypes score within 10% of optimal (balanced, top-heavy, spread)
- [ ] Backtest: prices generated from pre-S1 data, then scored against actual S1 results — correlation between price and actual PPG > 0.40

### Walk-forward validation
- Window 1: Generate prices for 2025 S1 from pre-S1 data, compare to actual Game Start VP
- Window 2: Generate prices for 2025 S2, compare
- Window 3: Generate prices for 2026 S1, compare to manual prices

---

## Decision Log

| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Quantile mapping over VORP | VORP was unstable with our data (collapsed to 4 unique prices). Quantile mapping guarantees spread. | Kei |
| 2026-04-06 | Small pickrate premium (2-8%) | Pickrate→performance r=0.11 (weak), but adds game-design value | Data |
| 2026-04-06 | No variance/ceiling premium | Variance not persistent (r=0.08 between stages per empirical analysis) | Data |
| 2026-04-06 | Calibrate quantile anchors against optimizer | Distribution must enable multiple archetypes, not just price accurately | Design |
| 2026-04-06 | Price range 5-15 VP (not 6-15) | User requirement — wider range gives more granularity | Kei |

---

## Related
- `specs/expected-points-model.md` — produces the input to this pipeline
- `specs/team-optimizer.md` — consumes prices, validates archetype diversity
- `empirical_analysis.py` — data analysis backing the findings
