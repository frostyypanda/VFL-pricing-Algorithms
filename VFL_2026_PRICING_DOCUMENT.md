# VFL 2026 Stage 1 — Pricing Algorithm

## 1. What This Is

A pricing algorithm for VFL 2026 Stage 1 that generates prices for 180 players across
Americas, EMEA, and Pacific (China excluded). Two price sets: **Manual** (human-set)
and **Generated** (pure algo with no manual overrides).

## 2. How It Works

**Training Data**: All VCT match data from 2024, 2025, and 2026 Kickoff + Santiago.
~8000+ played games across 400+ players.

**Core Approach**: Recency-weighted performance analysis with S-curve distribution shaping.
Recent events weighted higher. Players with less 2026 data lean more on historical data.
Single tournament spikes dampened.

**Factors**:

| Factor | Effective Weight | What it does |
|---|---|---|
| Performance (PPM) | ~55% | Points per map, recency-weighted |
| Team Strength | 12% | Win rate boost/penalty |
| Pickrate Sentiment | 10-40% | Community pick rates (higher weight for low-data players) |
| Team Brand | 5% | Fan-favorite premium (PRX, SEN, etc.) |
| Opponent Schedule | 7% | Actual 2026 Stage 1 matchups |
| Consistency | 3% | Reliable scorers get a small premium |

**Key Features**:
- Adaptive recency: less 2026 data = more weight on 2024/2025 history
- Outlier dampening: single tournament spikes kept at 60% of deviation
- Potential floor: budget players with one great run get a bump for upside
- Role adjustments: duelists -5% (naturally score more kills), sentinels +5%
- Boosted pickrate: community sentiment has high influence, especially for limited-data players
- Newcomer handling: 0-game players use team average PPM; hyped tier-2 newcomers get extra boost

## 3. Performance vs Manual

| Metric | Value |
|---|---|
| Correlation | 0.755 |
| MAE | 1.36 VP |
| Within 1.0 VP | 96/180 (53%) |
| Within 2.0 VP | 140/180 (78%) |
| Generated mean | 9.08 VP (target: 9.09) |
| Generated median | 9.00 VP |
| Manual mean | 9.09 VP |

## 4. Price Comparison (Top 30 by Generated VP)

| Player | Team | Pos | Manual | Generated | Diff |
|---|---|---|---|---|---|
| Marteen | Gentle Mates | D | 15.0 | 14.5 | -0.5 |
| jawgemo | G2 Esports | D | 13.5 | 14.5 | +1.0 |
| t3xture | Gen.G | D | 13.5 | 14.5 | +1.0 |
| Alfajer | FNATIC | S | 13.5 | 14.0 | +0.5 |
| leaf | G2 Esports | S | 12.5 | 14.0 | +1.5 |
| Dambi | Nongshim RedForce | D | 15.0 | 14.0 | -1.0 |
| Kaajak | FNATIC | D | 14.5 | 14.0 | -0.5 |
| Keiko | NRG Esports | C | 12.5 | 14.0 | +1.5 |
| trent | G2 Esports | I | 11.0 | 13.5 | +2.5 |
| aspas | MIBR | D | 14.5 | 13.5 | -1.0 |
| mada | NRG Esports | D | 14.5 | 13.5 | -1.0 |
| f0rsaken | Paper Rex | C | 13.5 | 13.5 | +0.0 |
| Zekken | MIBR | C | 12.5 | 13.5 | +1.0 |
| Chronicle | Team Vitality | S | 11.5 | 13.0 | +1.5 |
| Cryocells | 100 Thieves | S | 10.5 | 13.0 | +2.5 |
| skuba | NRG Esports | S | 9.5 | 13.0 | +3.5 |
| valyn | G2 Esports | C | 12.5 | 13.0 | +0.5 |
| Karon | Gen.G | S | 10.0 | 12.5 | +2.5 |
| Ivy | Nongshim RedForce | S | 10.0 | 12.5 | +2.5 |
| Francis | Nongshim RedForce | D | 12.0 | 12.5 | +0.5 |
| Meteor | T1 | S | 12.0 | 12.0 | +0.0 |
| d4v41 | Paper Rex | S | 11.0 | 12.0 | +1.0 |
| Demon1 | ENVY | S | 10.0 | 12.0 | +2.0 |
| Derke | Team Vitality | D | 14.0 | 12.0 | -2.0 |
| Jinggg | Paper Rex | D | 13.5 | 12.0 | -1.5 |
| OXY | Cloud9 | D | 13.0 | 12.0 | -1.0 |
| BuZz | T1 | D | 12.0 | 12.0 | +0.0 |
| BABYBAY | G2 Esports | D | 11.5 | 12.0 | +0.5 |
| nAts | Team Liquid | S | 11.0 | 11.5 | +0.5 |
| Minny | Gentle Mates | S | 10.5 | 11.5 | +1.0 |

## 5. Limitations

- Can't predict roster changes, role swaps, or coaching decisions
- Players with <5 games have uncertain prices (flagged)
- 0.5 VP granularity means some players will always be slightly off
- Past pickrate reflects past context, not necessarily future value
- New players (0 games) estimated from team context only

## 6. Uncertainty Flags

**HIGH Uncertainty**:

| Player | Team | Generated VP | Games |
|---|---|---|---|
| Absol | ZETA DIVISION | 6.0 | 4 |
| C1ndeR | VARREL | 5.5 | 3 |
| Chloric | Natus Vincere | 7.0 | 0 |
| DanteDeu5 | KRU Esports | 6.0 | 6 |
| Darker | LOUD | 6.5 | 3 |
| Erde | LOUD | 6.5 | 0 |
| ExiT | Natus Vincere | 6.5 | 0 |
| Favian | Eternal Fire | 6.0 | 4 |
| Izzy | Eternal Fire | 5.0 | 0 |
| Jerrwin | Sentinels | 8.5 | 0 |
| Krostaly | FUT Esports | 8.5 | 3 |
| NINJA | PCIFIC Esports | 6.0 | 3 |
| Neon | LEVIATAN | 8.5 | 3 |
| Reduxx | Sentinels | 9.0 | 4 |
| Rimuru/Zeus | Team Secret | 5.5 | 0 |
| Rosé | BBL Esports | 8.5 | 0 |
| Sayonara | Team Vitality | 8.5 | 0 |
| XuNa | VARREL | 7.0 | 3 |
| Zexy | VARREL | 6.5 | 3 |
| ZynX | Gen.G | 8.0 | 4 |
| al0rante | PCIFIC Esports | 7.5 | 3 |
| audaz | Eternal Fire | 7.0 | 4 |
| echo | Eternal Fire | 5.0 | 0 |
| eko | ZETA DIVISION | 6.0 | 4 |
| nekky | Eternal Fire | 6.5 | 4 |
| oonzmlp | VARREL | 6.0 | 3 |
| qpert | PCIFIC Esports | 5.5 | 3 |
| seven | PCIFIC Esports | 7.0 | 3 |

**MEDIUM Uncertainty**:

| Player | Team | Generated VP | Games |
|---|---|---|---|
| (none) | | | |\n
---

*Generated 2026-03-28 by VFL Pricing Algorithm v3.1*
