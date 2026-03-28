# VFL Pricing Algorithms — Summary for Stakeholders

## What We Built

We developed 6 different pricing algorithms (plus a machine learning approach) to generate fairer VFL player prices. Each was tested by predicting prices for 2025 Stage 1 and Stage 2, then checking how well those prices would have worked in practice.

---

## Our Four Goals (from the VFL Pricing Philosophy)

| # | Goal | What it means |
|---|---|---|
| 1 | **Accurate Valuation** | Prices should reflect how many points a player actually scores |
| 2 | **Competitive Compositions** | Multiple strong teams should be possible within the 100 VP budget |
| 3 | **Multiple Archetypes** | Different team-building strategies should all be viable (balanced, top-heavy, stars-and-scrubs, etc.) |
| 4 | **Perceived Value** | Every player should feel worth picking at their price — no obvious traps or freebies |

---

## The Algorithms — Pros & Cons

### 1. Baseline EMA (Exponential Moving Average)

**What it does:** Looks at recent performance (points per map) with more weight on newer games. Simple formula: better recent stats = higher price.

| Pros | Cons |
|---|---|
| Most intuitive — prices directly reflect recent form | Median price (9.45) drifts above the ideal 9.09 average |
| Good accuracy (prices within 1 VP of actual 51% of the time) | Produces mostly Top-Heavy team builds |
| Produces 3 different team archetypes in Stage 1 | Doesn't account for team popularity or matchup difficulty |
| Easy to explain and verify | Players on weak teams can be overpriced if they had one good event |

**Best for:** A simple, defensible starting point that gets you 80% of the way there.

---

### 2. Distribution-Aware

**What it does:** Same performance data as Baseline, but forces the price distribution into a healthier shape (more cheap options, less clustering in the middle).

| Pros | Cons |
|---|---|
| Closest median (8.92) to ideal 9.09 | Slightly less accurate than Baseline (more bold pricing choices) |
| Creates more budget options (31% of players under 8 VP) | Can underprice mid-tier players |
| Enables Stars-and-Scrubs builds alongside Top-Heavy | Still only 2 archetypes in optimal teams |
| Best average team performance in Stage 1 (810 pts) | More players at minimum price (6 VP) — some may feel "dumped" |

**Best for:** Maximizing team-building variety and enabling different strategies.

---

### 3. Role-Adjusted

**What it does:** Recognizes that Duelists naturally score more kills (and therefore more points) than supports/controllers. Adjusts prices so Duelists aren't automatically the best value picks.

| Pros | Cons |
|---|---|
| Produces the best single optimal team (951 pts Stage 1) | Lower accuracy — prices diverge more from "expected" |
| Fairer across roles — supports aren't always bargains | Role detection is approximate (no official role data yet) |
| 2 archetypes (Top-Heavy + Mixed) | Median (9.30) still above ideal |
| Helps perceived value — supports feel worth their price | Can misjudge flex players who switch roles |

**Best for:** Fairness across player roles. Will improve significantly when we get actual role data.

---

### 4. Combined (All Factors)

**What it does:** Blends everything — performance, team strength, player popularity, fan-favorite team bias, opponent difficulty, and consistency — into one price.

| Pros | Cons |
|---|---|
| Most holistic — considers factors others miss | Most complex and hardest to explain |
| Popular players (from pick data) cost more — supply/demand | Lower accuracy (intentionally deviates from current VFL pricing) |
| Fan-favorite teams (PRX, SEN, FNATIC) get popularity premium | Median 8.97 is closest to ideal but still slightly below |
| Accounts for roster changes (new players = less reliable team factor) | Pickrate data only available from 5 past events |
| New players flagged as "HIGH uncertainty" for manual review | |

**Best for:** A production system where you want one algorithm that considers everything.

---

### 5. Neural Network (Machine Learning)

**What it does:** A small AI model that learns the relationship between all player stats and their actual price. Outputs a price adjustment (-3 to +6 VP) on top of the 9.09 VP average, rounded to nearest 0.5.

| Pros | Cons |
|---|---|
| **By far the most accurate** — MAE 0.69 vs 1.2+ for others | Prices are rounded to 0.5 steps (less granular) |
| 76% of prices within 1 VP of actual (vs 51% for Baseline) | Correlation 0.85 (vs 0.69 for Baseline) — huge improvement | "Black box" — harder to explain why a specific player got a specific price |
| Produces Balanced + Mixed archetypes (healthiest mix) | Tighter price spread (std 1.5) — less dramatic price differences |
| 27 unique players across top 30 teams (best diversity) | Needs actual price data to train on — can't price a brand-new event from scratch |
| Learns non-linear patterns that formula-based approaches miss | May overfit to past pricing decisions (if those were flawed) |

**Best for:** Accuracy-first pricing where you trust the model and verify edge cases manually.

---

## Head-to-Head Results

### Stage 1 Backtest (trained on 2024 + early 2025, predicting Stage 1)

| Algorithm | Accuracy | Median VP | Best Team | Team Diversity | Archetypes | Overall Score |
|---|---|---|---|---|---|---|
| **Neural Network** | 0.69 MAE, 76% within 1VP | 9.25 | 1156 pts | 27 players | Mixed + Balanced | **90.5/100** |
| **Baseline EMA** | 1.20 MAE, 51% within 1VP | 9.45 | 1160 pts | 23 players | Top-Heavy + Balanced + Mixed | **88.4/100** |
| Distribution-Aware | 1.23 MAE, 53% within 1VP | 8.92 | 1233 pts | 19 players | Top-Heavy + Mixed | 83.7/100 |
| Combined | 1.44 MAE, 43% within 1VP | 8.96 | 1202 pts | 22 players | Top-Heavy + Mixed | 81.7/100 |
| Role-Adjusted | 1.33 MAE, 44% within 1VP | 9.30 | 1215 pts | 23 players | Mixed + Balanced | 80.3/100 |

### Stage 2 Backtest (trained on all data through Stage 1, predicting Stage 2)

| Algorithm | Accuracy | Median VP | Best Team | Overall Score |
|---|---|---|---|---|
| **Neural Network** | 0.66 MAE, 76% within 1VP | 9.00 | 1237 pts | **90.0/100** |
| **Baseline EMA** | 1.10 MAE, 51% within 1VP | 9.25 | 1238 pts | **83.2/100** |
| Role-Adjusted | 1.23 MAE, 47% within 1VP | 9.30 | 1221 pts | 81.6/100 |
| Distribution-Aware | 1.32 MAE, 45% within 1VP | 8.81 | 1252 pts | 80.7/100 |
| Combined | 1.43 MAE, 41% within 1VP | 8.79 | 1255 pts | 79.7/100 |

---

## What's Working Well

- **All algorithms beat random team-building by 40-60%** — the pricing creates meaningful strategic choices
- **High-uncertainty players are flagged** — new players, roster changes, and small sample sizes are called out for manual review
- **Team roster changes are accounted for** — teams with 2-3 new players get discounted team strength
- **Fan-favorite bias is captured** — popular teams like Paper Rex and Sentinels have a small popularity premium
- **Neural Network is remarkably accurate** — it correctly prices 3 out of 4 players within 1 VP

## What Still Needs Work

| Issue | Impact | Proposed Fix |
|---|---|---|
| **Only 2-3 archetypes emerge** from optimal teams | Top-heavy or stars-and-scrubs dominates, balanced teams rarely competitive | Compress the elite tier (cap max VP at 13-14) to reduce the advantage of having one superstar |
| **Team overlap is high** (~60%) | The same 15-20 "must-pick" players appear in most optimal teams | Better mid-tier differentiation — make more players "interesting" at their price point |
| **30% of players are HIGH uncertainty** | Manual review needed for ~60 players each stage | Will improve as we collect more data; pickrate data helps |
| **No IGL rate data yet** | Can't price the "captain premium" properly | Waiting for IGL pick data — will integrate when available |
| **No actual role data** | Role-Adjusted algorithm uses approximations | Need official role classifications for each player |
| **Opponent schedule not used** | Group stage matchups are known in advance but not factored into per-week pricing | Future: per-gameweek price adjustments based on opponent |

---

## Recommendation

**For immediate use:** Start with **Baseline EMA** as the foundation — it's simple, accurate, and explainable. Use **Neural Network** prices as a second opinion to catch cases where the formula gets it wrong.

**For the best overall experience:** Use **Combined** algorithm with manual override for HIGH uncertainty players. It's the most holistic and produces the fairest distribution (median closest to 9.09).

**Long-term:** Once we have role data and IGL pick rates, the **Neural Network** approach will likely become the primary algorithm, with distribution-shaping applied as a post-processing step to ensure healthy price spreads.

---

## Key Numbers to Remember

- **100 VP budget / 11 players = 9.09 VP average** — this is the mathematical ideal median
- **Price range: 6-15 VP** — every player costs between 6 and 15
- **Max 2 players per VCT team** — can't stack one team
- **IGL doubles one player's points** — makes high-ceiling players extra valuable
- **180 players in the non-China pool** — that's who we're pricing
