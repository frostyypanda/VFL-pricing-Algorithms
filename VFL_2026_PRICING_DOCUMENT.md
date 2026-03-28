# VFL 2026 Stage 1 — Pricing Algorithm

## 1. What This Is

We built a pricing algorithm for VFL 2026 Stage 1. It generates prices for ~170 players across Americas, EMEA, and Pacific (China excluded). Three price sets: **Manual** (human-set), **Generated** (pure algo), and **Generated + User** (algo + human corrections — the recommended set).

## 2. How It Works

**Training Data**: All VCT match data from 2024 season, 2025 season, and 2026 Kickoff + Santiago. ~8000+ played games across 400+ players.

**Ensemble Model**: We run 100 bootstrap iterations, each randomly weighting past tournaments differently (Bayesian bootstrap). Recent events weighted higher. Average all 100 to get a robust price per player.

**Factors**:

| Factor | Weight | What it does |
|---|---|---|
| Performance (PPM) | 60% | Points per map, recency-weighted |
| Team Strength | 12% | Win rate boost/penalty |
| Pickrate | 8% | Community sentiment from past VFL picks |
| Team Brand | 8% | Fan-favorite premium (PRX, SEN, etc.) |
| Opponent Schedule | 7% | Actual 2026 Stage 1 matchups |
| Consistency | 5% | Reliable scorers get a small premium |

**Key Improvements Over Simple EMA**:
- Adaptive recency: players with less 2026 data lean more on 2024/2025 history
- Single tournament spikes dampened: one great event doesn't = max price
- Potential floor: if a budget player had one great run, they get a bump for upside
- Role adjustments: duelists discounted 5% (naturally score more kills), sentinels boosted 5%
- Pickrate sentiment: less data = more weight on past community picks

**Generated + User**: Where algo and manual disagree by 2+ VP, human corrections are applied (roster changes, role swaps, context the algo can't see). 31 specific player overrides. This is the recommended price set.

## 3. How Well It Meets Our Criteria

**1. Accurate Valuation** (target: ~80%)
- Gen+User prices correlate with past performance (r=0.66)
- Mean price: 9.11 VP (target 9.09)
- Median: 9.00 VP
- Distribution shaped via S-curve to avoid mid-tier clustering

**2. Competitive Compositions** (target: ~50%)
- Budget math allows multiple team archetypes within 100 VP
- Top-30 teams use 20-27 unique players (not all the same squad)
- Excluding auto-picks costs only 3-5% points

**3. Multiple Archetypes** (target: currently poorly met -> improved)
- Price distribution enables: Top-Heavy, Balanced, Mid-Heavy, Dual-Star, Spread
- S-curve shaping creates meaningful gaps between tiers
- No single dominant strategy

**4. Perceived Value** (target: currently poorly met -> improved)
- Mean 9.11, median 9.00 — close to ideal 9.09 average
- Value equity across tiers improved (Pts/VP roughly equal at each tier)
- 24 players flagged for manual review (uncertainty flags)

## 4. What It Can't Do

- Can't predict roster changes, role swaps, or coaching decisions
- Players with <5 games have uncertain prices (flagged for manual review)
- 0.5 VP granularity means some players will always be slightly over/underpriced
- Past pickrate reflects past context, not necessarily future value

## 5. Uncertainty Flags

24 players are flagged as HIGH or MEDIUM uncertainty. These should be manually reviewed before finalizing prices.

**HIGH Uncertainty** (fewer than 5 games or no 2026 data):

| Player | Team | Gen+User VP |
|---|---|---|
| Absol | ZETA DIVISION | 8.5 |
| Autumn - AUS | Global Esports | 8.5 |
| C1ndeR | VARREL | 5.5 |
| Dantedeu5 | KRU Esports | 10.0 |
| Darker | LOUD | 6.5 |
| Demur | Eternal Fire | 7.0 |
| Favian | Eternal Fire | 6.5 |
| Inspire | ENVY | 7.0 |
| KROSTALY | FUT Esports | 7.0 |
| Kyu | Sentinels | 7.5 |
| NINJA | PCIFIC Esports | 6.0 |
| Reduxx | Sentinels | 9.0 |
| Rose | BBL Esports | 6.5 |
| XuNa | VARREL | 7.0 |
| Zexy | VARREL | 9.5 |
| ZynX | Gen.G | 8.5 |
| al0rante | PCIFIC Esports | 7.5 |
| audaz | Eternal Fire | 7.0 |
| eko | ZETA DIVISION | 8.0 |
| nekky | Eternal Fire | 6.5 |
| oonzmlp | VARREL | 7.5 |
| qpert | PCIFIC Esports | 5.5 |
| seven | PCIFIC Esports | 8.0 |

**MEDIUM Uncertainty**: Pxs (LEVIATAN, 9.0 VP)

---

*Generated 2026-03-28 by VFL Pricing Algorithm v3.0*
