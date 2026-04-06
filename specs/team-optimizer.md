# Feature: Optimal Team Builder & Expected Points Engine

## Status
draft

## Why this exists

We need a rigorous, fully automated system that answers: **"Given player prices, what is the best possible VFL team across all 6 gameweeks of a stage?"**

This serves two purposes:
1. **Direct use**: Generate optimal team recommendations for any price list (manual or algorithmic).
2. **Pricing validation**: By computing expected points and building optimal teams under different price sets, we can evaluate whether a pricing algorithm produces healthy game dynamics (archetype diversity, competitive compositions, no dominant strategy).

The current implementation (`vfl_2026_final.py`) uses a randomized greedy heuristic with noise injection. This is fast but non-optimal — it misses the true best team, especially for the multi-week transfer problem where short-term greedy decisions conflict with long-term GW6 positioning. A ground-up rebuild should use principled optimization (ILP) and a proper expected points model.

---

## What it does

### Two modes of operation

**Mode 1 — Build from scratch (GW1)**
Given a player pool with prices, roles, teams, and expected points per gameweek, find the 11-player squad that maximizes total expected points across all 6 gameweeks, subject to all constraints.

**Mode 2 — Transfer optimization (GW2-6)**
Given an existing squad with locked slot assignments, find the best 0–3 transfers per gameweek to maximize remaining expected points, subject to all constraints including position-locked transfers.

### Outputs

For each run, the system produces:

| Output | Description |
|--------|-------------|
| **Optimal squad per GW** | 11 players with slot assignments (D1, D2, I1, I2, C1, C2, S1, S2, W1, W2, W3) |
| **Transfers per GW** | Which players are swapped in/out and in which slot (GW2-6) |
| **IGL per GW** | Which player is designated captain (2x points) each gameweek |
| **Expected points per player per GW** | Full 180-player × 6-GW matrix of expected fantasy points |
| **Suggested price per player** | Algorithmic price derived from the expected points model (5–15 VP, 0.5 increments) |
| **Season total expected points** | Sum across all 6 GWs including IGL bonus |

---

## Rules & Constraints

### Squad Composition
- **11 players** total
- **Role slots**: 2 Duelist, 2 Controller, 2 Initiator, 2 Sentinel (8 slots)
- **Wildcard slots**: 3 (any role)
- **Budget**: Total squad price must not exceed **100 VP**
- **Team limit**: Maximum **2 players** from any single VCT team
- **Price range**: Player prices are **5.0–15.0 VP** in **0.5 VP increments**

### Gameweek Schedule (Stage 1, 2026)

| GW | Regions Playing | Notes |
|----|----------------|-------|
| 1 | EMEA, Pacific | Americas does NOT play — their players score 0 |
| 2 | EMEA, Pacific, Americas | Americas W1 |
| 3 | EMEA, Pacific, Americas | Americas W2 |
| 4 | EMEA, Pacific, Americas | Americas W3 |
| 5 | EMEA, Pacific, Americas | Americas W4 |
| 6 | **Americas only** | EMEA and Pacific do NOT play — their players score 0 |

Americas starts 1 week late and ends 1 week late. This is the single most important strategic constraint: **by GW6, only Americas players score points**, so the squad must transition toward Americas players over the season.

### Transfers
- **3 transfers per gameweek** (GW2 onward)
- **Position-locked**: A player in a role slot (D/C/I/S) can only be replaced by a player of the **same role**. A player in a wildcard slot (W) can be replaced by a player of **any role**.
- Transfers do not cost points.
- Unused transfers cannot be banked/rolled over.
- Slot assignments are permanent from GW1 — you cannot reassign a slot's role type.

### IGL (Captain)
- Each gameweek, designate **1 player as IGL**
- That player's points are **doubled** (added again on top)
- IGL should be the highest-expected-points player each GW
- If the IGL's team doesn't play that GW, the doubling is wasted (scores 0 × 2 = 0)

### Scoring System

Points are earned per game (one game per gameweek per player). A game consists of 2–3 maps.

**Kill Points (per map)**

| Kills | Points |
|-------|--------|
| 0 | -3 |
| 1–4 | -1 |
| 5–9 | 0 |
| 10–14 | +1 |
| 15–19 | +2 |
| 20–24 | +3 |
| 25–29 | +4 |
| 30+ | +1 per additional 5 kills |

**Multi-Kill Bonuses (per round)**

| Event | Points |
|-------|--------|
| 4K in a round | +1 |
| 5K (ace) | +3 |
| 6K | +5 |
| 7K | +10 |

**Map Win/Loss Points (team-based, all players on the team)**

| Result | Points |
|--------|--------|
| Map win | +1 |
| Win by 5–9 rounds | +1 bonus |
| Win by 10+ rounds | +2 bonus |
| Loss by 10+ rounds | -1 |

**Series Bonuses (team-based)**

| Result | Points |
|--------|--------|
| 2-0 series win | +2 |
| 2-1 series win | +1 |

**VLR Rating Bonuses (per game, across all maps)**

| Condition | Points |
|-----------|--------|
| Highest average VLR rating in the match | +3 |
| 2nd highest | +2 |
| 3rd highest | +1 |
| Average VLR rating ≥ 2.0 | +3 |
| Average VLR rating ≥ 1.75 | +2 |
| Average VLR rating ≥ 1.5 | +1 |

Note: Rating bonuses are across all 10 players in the match (both teams), not just one team.

---

## Architecture

### Component 1: Expected Points Model

**See `specs/expected-points-model.md` for the full spec.**

Summary: Uses recency-weighted historical Pts (not PPM, not decomposed components) with empirical Bayes shrinkage and RidgeCV-learned feature weights. Every parameter is learned from data via walk-forward cross-validation. Opponent adjustment is a single learned coefficient.

The model produces `E[pts_i_gw]` for each player `i` and gameweek `gw`. Players whose team doesn't play that GW get 0.

#### Output
A matrix: `expected_pts[player][gw]` for all ~180 players × 6 GWs.

---

### Component 2: Suggested Price Derivation

Given the expected points matrix, derive a suggested price for each player.

#### Approach: VORP (Value Over Replacement Player)

1. **Compute season value**: For each player, sum expected points across all GWs they're likely to be rostered. Weight by roster probability (a player who only scores in GW1-5 but not GW6 is less valuable than one who scores GW2-6).

2. **Define replacement level**: The marginal player — the worst player you'd ever roster. Approximately the value of the ~180th ranked player (or the cheapest viable fill at the minimum price).

3. **Compute VORP**: `VORP_i = season_value_i - replacement_value`

4. **Convert to VP**:
   ```
   raw_VP_i = VP_MIN + (total_budget_available / total_VORP) * VORP_i
   ```
   Where `total_budget_available = BUDGET - (VP_MIN * SQUAD_SIZE)` = `100 - (5.0 * 11)` = `45.0` VP of discretionary spending.

5. **Round**: Snap to nearest 0.5 VP increment.

6. **Clip**: Enforce [5.0, 15.0] range.

7. **Budget calibration**: Iteratively adjust so that the mean price is close to 9.09 VP (100/11) and the top ~15 optimal teams all sum to ≤ 100 VP with meaningful tradeoffs.

---

### Component 3: Team Optimizer

#### From-scratch optimizer (GW1)

This is a constrained combinatorial optimization problem. Use **Integer Linear Programming (ILP)**.

**Decision variables:**
- `x[i]` ∈ {0, 1} — whether player `i` is selected
- `d[i]`, `c[i]`, `init[i]`, `s[i]` ∈ {0, 1} — whether player `i` fills a D/C/I/S role slot
- `w[i]` ∈ {0, 1} — whether player `i` fills a wildcard slot
- `igl[i][gw]` ∈ {0, 1} — whether player `i` is IGL in gameweek `gw`
- `transfer_in[i][gw]`, `transfer_out[i][gw]` ∈ {0, 1} — transfer decisions

**Objective: Maximize total expected points across all 6 GWs including IGL bonus.**

For GW1-only (from-scratch, no transfers):
```
maximize Σ_i x[i] * E[pts_i_gw1] + Σ_i igl[i] * E[pts_i_gw1]
```

For full-season optimization (if computationally feasible):
```
maximize Σ_gw Σ_i roster[i][gw] * E[pts_i_gw] + Σ_gw Σ_i igl[i][gw] * E[pts_i_gw]
```

**Constraints:**
```
Σ_i x[i] = 11                              # exactly 11 players
Σ_i x[i] * price[i] ≤ 100                  # budget
Σ_{i ∈ team_t} x[i] ≤ 2  ∀ teams t         # max 2 per team
Σ_i d[i] = 2, where d[i] ≤ x[i], d[i] ≤ is_duelist[i]     # exactly 2 duelists in D slots
Σ_i c[i] = 2, where c[i] ≤ x[i], c[i] ≤ is_controller[i]  # exactly 2 controllers in C slots
Σ_i init[i] = 2, where init[i] ≤ x[i], init[i] ≤ is_initiator[i]  # exactly 2 initiators in I slots
Σ_i s[i] = 2, where s[i] ≤ x[i], s[i] ≤ is_sentinel[i]    # exactly 2 sentinels in S slots
Σ_i w[i] = 3, where w[i] ≤ x[i]                            # exactly 3 wildcards
d[i] + c[i] + init[i] + s[i] + w[i] = x[i]  ∀ i            # each selected player in exactly 1 slot
Σ_i igl[i][gw] = 1  ∀ gw                    # exactly 1 IGL per GW
igl[i][gw] ≤ roster[i][gw]                  # IGL must be on roster
```

**Solver**: Use `scipy.optimize.milp`, Google OR-Tools (`cp_model`), or PuLP. OR-Tools recommended for flexibility and performance.

#### Transfer optimizer (GW2-6)

Given the current roster with slot assignments, find the best 0–3 transfers.

**Per-gameweek greedy approach (if full-season ILP is too complex):**

For each GW independently:
1. Enumerate all possible transfer combinations: 0 transfers (1 option), 1 transfer (11 out × valid replacements), 2 transfers, 3 transfers.
2. For each combination, check constraints (budget, team limit, role match).
3. Score = expected points for the resulting squad that GW (including IGL pick).
4. Pick the highest-scoring valid combination.

With 180 players and 11 slots, the 1-transfer space is ~11 × 180 = 1,980 options (minus invalids). The 3-transfer space is larger but still tractable with pruning.

**Full-season ILP approach (preferred, more complex):**

Model the entire 6-GW season as one optimization problem. Decision variables include roster state at each GW and transfer decisions between GWs. This finds the globally optimal strategy, including the critical "transition to Americas for GW6" planning.

**Transfer constraints in the ILP:**
```
roster[i][gw] - roster[i][gw-1] ≤ transfer_in[i][gw]       # new player = transfer in
roster[i][gw-1] - roster[i][gw] ≤ transfer_out[i][gw]      # removed player = transfer out
Σ_i transfer_in[i][gw] ≤ 3  ∀ gw ≥ 2                       # max 3 transfers per GW
Σ_i transfer_in[i][gw] = Σ_i transfer_out[i][gw]  ∀ gw     # transfers are swaps
```

**Position-lock constraints:**
```
# If player i is transferred into a role slot, they must have that role
# If player i is transferred into a wildcard slot, any role is fine

# Track which slot each player occupies across GWs
slot_d[i][gw], slot_c[i][gw], slot_i[i][gw], slot_s[i][gw], slot_w[i][gw] ∈ {0,1}

# A player entering a D slot must be a duelist
slot_d[i][gw] ≤ is_duelist[i]
# A player entering a C slot must be a controller
slot_c[i][gw] ≤ is_controller[i]
# etc.

# Slot counts are preserved across GWs (slots don't change type)
Σ_i slot_d[i][gw] = 2  ∀ gw
Σ_i slot_c[i][gw] = 2  ∀ gw
Σ_i slot_i[i][gw] = 2  ∀ gw  (using init instead of i to avoid confusion)
Σ_i slot_s[i][gw] = 2  ∀ gw
Σ_i slot_w[i][gw] = 3  ∀ gw
```

#### GW6 Americas-Only Handling

This is NOT a special case — it falls naturally out of the expected points model. In GW6, all non-Americas players have `E[pts] = 0`. The optimizer will naturally:
1. Prefer Americas players in wildcard slots (since wildcards are flexible and can be swapped for any role)
2. Start transitioning Americas players in during GW2-5 via transfers
3. By GW6, maximize Americas coverage within the 3-transfers-per-GW constraint

The critical strategic question the optimizer must solve: **How many Americas players to include from GW1 (where they score 0) to ease the GW6 transition?**

With 3 transfers/GW × 4 GWs (GW2-5) = 12 possible transfers, and 11 slots to fill, starting with some Americas players in GW1 (especially in wildcard slots where they're flexible) is likely optimal despite sacrificing GW1 points.

---

## Boundaries & Edge Cases

### What this intentionally does NOT do
- Does not handle international event formats (Kickoff, Masters, Champions) — stage play only
- Does not predict roster changes or benching mid-season — uses the known roster at time of computation
- Does not model player injuries, travel fatigue, or meta shifts during the stage
- Does not account for VFL manager psychology or market dynamics
- Does not handle playoff pricing (different format, different rules)

### Edge Cases

**Player with 0 expected games**: A player whose team is eliminated or who is benched for the entire stage. Set E[pts] = 0 for all GWs. The optimizer will never select them unless they're the cheapest filler needed to satisfy constraints.

**Tied expected points for IGL**: If two players have identical expected points in a GW, pick either. In practice, choose the one with higher variance (higher ceiling = better IGL pick since 2x amplifies upside).

**Budget exactly 100.0**: The optimizer should find teams that use as much of the 100 VP budget as possible. Leftover VP is wasted value.

**No valid team exists**: If constraints are so tight that no 11-player team satisfies all of them (extremely unlikely with 180 players and 5-15 VP range), report an error with the binding constraint.

**Player role ambiguity**: Some players may play multiple roles across different stages. Use their most recent/primary role as listed in the roster data.

**Americas player in GW1**: Scores 0. This is intentional and expected. The optimizer accounts for this.

**Non-Americas player in GW6**: Scores 0. The optimizer accounts for this. Some non-Americas players may remain on the roster in GW6 if the 3-transfer-per-GW limit prevents full transition — they simply contribute 0 points that week.

**Minimum-price squad**: 11 × 5.0 = 55 VP. This leaves 45 VP of "discretionary" budget. The optimizer distributes this across the highest-value upgrades.

---

## Testing & Verification

### Key Scenarios

- [ ] **GW1 from-scratch**: Given 2026 Stage 1 manual prices, build the optimal GW1 squad. Verify: exactly 11 players, budget ≤ 100, 2D/2C/2I/2S/3W, max 2 per team, no Americas players scoring 0 unless strategically justified.
- [ ] **Full 6-GW optimization**: Build optimal squad across all 6 GWs. Verify: ≤ 3 transfers per GW, position-locked transfers, budget always ≤ 100, GW6 squad is heavily or fully Americas.
- [ ] **Transfer mode**: Given a specific existing team, optimize transfers for GW3. Verify: ≤ 3 transfers, position-locked, budget maintained.
- [ ] **GW6 transition**: Verify the optimizer plans ahead — starts acquiring Americas players before GW6, doesn't get stuck with 8 non-Americas players and only 3 transfer slots.
- [ ] **IGL selection**: Verify IGL is always the highest-expected-points player each GW (or highest-variance if points are equal).
- [ ] **Expected points output**: Verify the full 180 × 6 matrix is produced, non-playing players have 0, values are reasonable (typical range: 0–25 pts per game).
- [ ] **Suggested prices output**: Verify all prices in [5.0, 15.0], 0.5 increments, mean close to 9.09.
- [ ] **Backtest against 2025 Stage 1**: Use 2024 + 2025 pre-Stage-1 data to predict expected points, build optimal team, score against actual 2025 Stage 1 results. Compare vs random teams and vs teams built with actual VFL prices.
- [ ] **Backtest against 2026 Stage 1**: Use all pre-Stage-1 data to predict, score against actual results.
- [ ] **Price range boundary**: Confirm prices at 5.0 and 15.0 exist (the full range is used, not clustered in the middle).

### Edge Cases
- Team with 3+ roster changes mid-season — verify algorithm uses current roster
- Player who played in 2024 but not 2025 or 2026 — heavy shrinkage, reasonable price
- Player with 0 historical games (brand new) — uses role+team prior, priced near median
- GW where a player's team has a bye (doesn't play that week) — scores 0, handled correctly
- All 12 Americas teams × 5 players = 60 Americas players — verify the optimizer can build a valid GW6 team from this pool within budget

### Automation Notes
- All scenarios should be automated as pytest tests
- Backtests should run end-to-end: data load → expected points → team optimization → scoring → comparison
- Use deterministic random seeds for reproducibility
- Performance benchmark: full 6-GW optimization should complete in < 60 seconds

---

## Decision Log

| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Price range 5–15 VP (not 6–15) | User requirement — wider range gives more pricing granularity at the low end | Kei |
| 2026-04-06 | Position-locked transfers | VFL rules — D→D, C→C, I→I, S→S, W→any | Kei |
| 2026-04-06 | ILP over greedy heuristic | Greedy misses globally optimal multi-week strategies, especially GW6 transition planning | — |
| 2026-04-06 | Use Pts not PPM, not decomposed | Pts r=0.54 vs PPM r=0.44. Decomposed model introduced bugs and underestimated by 28%. | Data |
| 2026-04-06 | Quantile-based pricing over VORP | VORP collapsed to 4 unique prices. Quantile mapping guarantees spread. | Data |

---

## Related
- `specs/expected-points-model.md` — the statistical model that produces E[pts]
- `specs/pricing-algorithm.md` — converts expected points to VP prices
- `empirical_analysis.py` — data analysis backing design decisions
- `CLAUDE.md` — VFL rules, scoring system, data documentation
- `schedule_2026.py` — GW schedule and matchup data

---

## Implementation Plan

### Phase 1: Data Pipeline
1. Clean data loader: standardize all 3 CSVs, filter China, normalize team names
2. Walk-forward data splitter: given target stage/year, return training data and actual results
3. Manual price loader for the 2026 pricing sheet
4. **Test**: Correct row counts, no China teams, column types right

### Phase 2: Expected Points Model (see `specs/expected-points-model.md`)
1. Empirical Bayes: estimate tau², sigma² from data, compute per-player shrunk Pts
2. Feature construction: avg_pts windows, team_win_rate, n_games, role
3. RidgeCV: learn feature weights via walk-forward + LOO
4. Opponent adjustment: learn beta from within-stage game-level data
5. Per-GW matrix: apply schedule, set non-playing teams to 0
6. Ensemble: average empirical Bayes, Ridge, and EMA models
7. **Test**: Walk-forward MAE < 2.5, correlation > 0.50, top-10 players plausible

### Phase 3: Pricing (see `specs/pricing-algorithm.md`)
1. Compute season value from expected points matrix
2. Apply small pickrate adjustment (learned weight)
3. Quantile mapping to VP [5.0, 15.0]
4. Budget calibration (mean → 9.09)
5. Distribution verification checks
6. **Test**: Mean ~9.09, full range used, correlation with manual > 0.65

### Phase 4: Team Optimizer (full-season ILP)
1. Single PuLP ILP: roster, slots, IGL, transfers across all 6 GWs
2. Both modes: from-scratch and transfer-from-existing
3. **Test**: All constraints satisfied, GW6 = 11/11 AMER, budget ≤ 100 every GW

### Phase 5: Backtesting & Validation
1. Walk-forward: predict 2025 S1 → score against actuals
2. Walk-forward: predict 2025 S2 → score against actuals  
3. Composition diversity: multiple archetypes within 10% of optimal
4. Compare prices to manual prices (correlation, MAE)

### Phase 6: New Player Handling
1. vlr.gg player stats scraper (profile page, 60d timespan)
2. Role inference from agent breakdown
3. Tier-2 deflation factor calibrated from historical promotees
4. Integrate as cold-start prior in empirical Bayes
5. **Test**: New player prices near median, high uncertainty flag
