# VFL Pricing Algorithms

## What is VFL?

**Valorant Fantasy League (VFL)** is a community-run fantasy game for the Valorant Champions Tour (VCT). Managers build squads of professional Valorant players and score points based on real match performances across the VCT season.

- Website: https://www.valorantfantasyleague.net
- Stats sourced from: https://www.vlr.gg

---

## Game Rules (Regular Stage Play — our focus)

### Squad Composition
- **11 players** total (no bench in 2026 split — all play every week)
  - 2 Duelists
  - 2 Controllers
  - 2 Initiators
  - 2 Sentinels
  - 3 Wildcards (any role)
- Max **2 players per VCT team**
- Previously had 2 bench slots (8 starters + 3 bench), now all 11 play

### Budget
- **100 VP** total budget
- Player prices range from **6 to 15 VP**
- Average price: **~9.09 VP** (100 / 11)

### IGL (Captain)
- You designate **1 player as IGL** each gameweek
- That player's points are **doubled**
- This makes having at least one high-ceiling player essential

### Regions Included
- **Americas** (12 teams): 100 Thieves, Cloud9, Evil Geniuses, FURIA, KRU Esports, Leviatan, LOUD, MIBR, NRG, Sentinels, The Guard / 2Game Esports, etc.
- **EMEA** (12 teams): Team Heretics, Fnatic, Team Liquid, FUT Esports, GIANTX, Karmine Corp, BBL Esports, KOI, Natus Vincere, Team Vitality, Apeks, Gentle Mates
- **Pacific** (12 teams): DRX, Gen.G, Global Esports, Paper Rex, Rex Regum Qeon, T1, Talon, Team Secret, Zeta Division, DetonatioN FocusMe, BOOM Esports, Nongshim RedForce
- **China** is **NOT** included
- Total: **36 partnered teams**, ~180 players in the VFL pool for non-China regions

### Format Context
- VCT season: Kickoff -> Masters 1 -> Stage 1 -> Masters 2 -> Stage 2 -> Champions
- VFL runs for **regular stages** (Stage 1, Stage 2) — 5 round-robin games per team in groups
- Kickoff and international events (Masters, Champions) use a **different VFL format** (not our primary focus but data is available)
- Stage format: 2 groups of 6, round-robin, then double-elim playoffs

---

## Scoring System

### Kill Thresholds (per map)
| Kills on map | Points |
|---|---|
| 0-9 | 0 |
| 10-14 | +1 |
| 15-19 | +2 |
| 20-24 | +3 |
| 25-29 | +4 |
| 30+ | +1 per additional 5 kills |

### Multi-Kill Bonuses (per round)
| Event | Points |
|---|---|
| 4K in a round | +1 |
| 5K (ace) in a round | +3 |

### Map Win/Loss Points (team-based)
| Result | Points |
|---|---|
| Map win | +1 |
| Win by 5+ rounds (e.g. 13-8) | +1 bonus |
| Win by 10+ rounds (e.g. 13-3) | +2 bonus |
| Loss 13-3 or worse | -1 |

### VLR Rating Bonuses (per game, across all maps)
| Condition | Points |
|---|---|
| Top VLR rating in the game | +3 |
| 2nd best VLR rating | +2 |
| 3rd best VLR rating | +1 |
| VLR rating above 2.0 | +3 |
| VLR rating above 1.75 | +2 |
| VLR rating above 1.5 | +1 |

### IGL Multiplier
- Selected IGL gets **2x all points** for that gameweek

---

## Available Data

### CSV Files
Two CSV files with per-game, per-player scoring data:

- **`2025 VFL.csv`** — 2025 VCT season (18,618 rows, 301 players, 48 teams including China)
- **`2026 VFL.csv`** — 2026 VCT season (18,951 rows, 250 players, 48 teams including China)

### CSV Columns
| Column | Description |
|---|---|
| Col 1 (unnamed) | Team name |
| Col 2 (unnamed) | Player name |
| Stage | Tournament stage (Kickoff, Stage 1, Stage 2, bangkok, Toronto, Champions, etc.) |
| Wk | Week number (W1, W2, W3) |
| Game | Game identifier (G1-G5 for stage, UR1/UQF/USF/UF/LR1-4/LF/GF for playoffs, SR1-3 for swiss) |
| Pts | Total fantasy points scored |
| T.Pts | Team-based points (map wins/losses) |
| P.Pts | Personal points (kills, multi-kills, rating) |
| PPM | Points per map |
| Adj.VP | Adjusted VP (dynamically calculated price) |
| P? | Played flag (1 = played, 0 = did not play) |
| 0k-50k | Kill bracket counts per map (how many maps the player hit each kill threshold) |
| 4ks, 5ks | Number of 4K and 5K rounds |
| TOP3/TOP2/TOP1 | Whether player was top 3/2/1 VLR rating in the game |
| 1.5R2/1.75R2/2.0R2 | Whether player achieved 1.5+/1.75+/2.0+ VLR rating |
| PR Avg. | Running average points (performance rating) |
| W/L | Match result (W/L) |
| Game Start VP / Game End VP | Player's VP price before and after the game |

### Image
- **`vfl price stats.webp`** — Bar chart showing price distribution across multiple splits/stages (4 colored series). Shows player count at each 0.5 VP price point from 5 to 15. Key observation: heavy clustering around 7-8.5 VP, with the median consistently below 9 VP.

---

## Project Goals

### Primary Objective
Build pricing algorithms that generate better VFL player price lists for regular stage play.

### Pricing Philosophy & Design Goals

1. **Accurate valuation (currently ~80% met):** Good spread of prices that accurately represent the mean/median value of a player based on historic data with a gentle bias toward recency.

2. **Competitive compositions (currently ~50% met):** Prices should allow multiple competitive potential compositions of players to score good points — not just one dominant strategy.

3. **Multiple archetypes (currently poorly met):** Enable different viable team-building strategies (top-heavy, balanced, spread, etc.) rather than forcing everyone into one approach.

4. **Perceived value (currently poorly met):** Every player should feel worth considering at their price — their price makes them perceived to be valued in some sense.

### Known Problems with Current Pricing
- Median prices are always below 9 VP (the mathematical average), indicating a skewed distribution
- Mini-peaks at 11-13 VP do more harm than good — they push the meta toward top-heavy being the only viable strategy
- Leaderboards end up dominated by top-heavy teams that luck into their cheap picks being overperformers
- Current approach: accurately value good players first, then try to balance the spread after the fact (which doesn't work well enough)

### Approach
- Build multiple different algorithms to explore different strategies
- Use recency-weighted historical data (more weight on recent performances)
- Test algorithms by backtesting:
  - Predict Stage 2 2025 prices using Stage 1 2025 + all 2024 data
  - Predict Kickoff prices using prior year data
- Consider fetching 2024 data from vlr.gg to supplement the CSV data
- May convert CSVs to JSON if easier to work with

### Success Criteria
- Backtest accuracy against actual VFL prices
- Composition diversity: do the generated prices allow many different viable 100 VP teams?
- Distribution analysis: is the spread healthy or does it force a single dominant archetype?
- Price-to-performance correlation: do prices predict actual fantasy output?

---

## Technical Notes

- CSVs use latin-1 encoding (special characters in team names like LEVIATaN)
- Rows with `P? = 0` and all-zero stats indicate the player did not participate in that game (team eliminated, bye week, etc.)
- The `Game Start VP` and `Game End VP` columns show how VFL's current dynamic pricing moves after each game
- China teams are in the data but NOT relevant for VFL pricing (filter them out)
- 2026 CSV has additional columns vs 2025: `6ks`, `7ks` multi-kill categories, and a `START` row per player

---

## Algorithms to Explore

*(To be expanded as we develop them)*

1. **Baseline EMA (Exponential Moving Average):** Recency-weighted average of PPM, map to 6-15 VP range
2. **Distribution-aware pricing:** Optimize for composition diversity by targeting a specific price distribution shape
3. **Role-adjusted pricing:** Account for role-based scoring differences (duelists naturally score higher)
4. **Team strength multiplier:** Factor in team win rates since team points are a significant scoring component
5. **Variance-aware pricing:** Consider consistency vs. volatility in player performance
