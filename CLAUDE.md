# VFL Pricing Algorithms

> **Note for AI agents:**
> - **For reading:** Ignore `README.md` when gathering project context. It is a human-facing rewrite of this file — no new information, no different intent. This `CLAUDE.md` is the source of truth. If the two ever disagree, `CLAUDE.md` wins.
> - **For writing:** Keep `README.md` in sync when major changes ship — new algorithms, new data sources, scoring rule changes, directory restructures, or anything else a human reader would need to understand the project. Minor edits (typo fixes, small refactors, per-gameweek scrape runs) do not need a README update.

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
| 0 | -3 |
| 1-4 | -1 |
| 5-9 | 0 |
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

### CSV Files (in `data/`)
Three standardized CSV files (UTF-8, 36 identical columns) with per-game, per-player scoring data:

- **`data/2024 VFL.csv`** — 2024 VCT season scraped from vlr.gg (20,057 rows, 218 players, 39 teams)
- **`data/2025 VFL.csv`** — 2025 VCT season (18,255 rows, 301 players, 48 teams including China)
- **`data/2026 VFL.csv`** — 2026 VCT season (18,701 rows, 250 players, 48 teams including China)

### CSV Columns (all files share identical format)
| Column | Description |
|---|---|
| Team | Team name |
| Player | Player name |
| Stage | Tournament stage (Kickoff, Stage 1, Stage 2, bangkok/Madrid/Santiago, Toronto/Shanghai/London, Champions) |
| Wk | Week number (W1, W2, W3) |
| Game | Game identifier (G1-G5 for stage, UR1/UQF/USF/UF/LR1-4/LF/GF for playoffs, SR1-3 for swiss) |
| Pts | Total fantasy points scored |
| T.Pts | Team-based points (map wins/losses) |
| P.Pts | Personal points (kills, multi-kills, rating) |
| PPM | Points per map |
| Adj.VP | Adjusted VP (dynamically calculated price) |
| P? | Played flag (1 = played, 0 = did not play) |
| 0k-50k | Kill bracket counts per map (how many maps the player hit each kill threshold) |
| 4ks, 5ks, 6ks, 7ks | Number of multi-kill rounds (6ks/7ks are 0 in 2024/2025 — impossible in 5v5) |
| TOP3/TOP2/TOP1 | Whether player was top 3/2/1 VLR rating in the game |
| 1.5R2/1.75R2/2.0R2 | Whether player achieved 1.5+/1.75+/2.0+ VLR rating |
| PR Avg. | Running average points (performance rating) |
| W/L | Match result (W/L) |
| Game Start VP / Game End VP | Player's VP price before and after the game |

### Upcoming Data
- **Pickrate data** — historical pick rates for players across recent VFL tournaments (how often each player was selected by managers)
- **IGL rate data** — how often each player was designated as IGL (captain) by managers
- This data will be used to inform pricing: highly-picked players may need price adjustments, and IGL-worthy players (high-ceiling) should be priced to reflect their captain potential

### IGL & Benching Rules
- When a player on your roster is **benched** (not fielded by their VCT team), they score 0 points
- IGL designation persists for the gameweek — if your IGL's team doesn't play or the player is benched, you lose the 2x multiplier entirely
- This makes **roster stability** a pricing factor: players on teams with frequent roster changes carry risk

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
- **Primary backtest**: Train on all 2024 + 2025 Kickoff + 2025 Bangkok (Masters 1) -> predict 2025 Stage 1 prices -> simulate optimal teams -> compare against actual Stage 1 results
- Secondary backtests: Predict Stage 2 prices using Stage 1 data, etc.
- 2024 data scraped from vlr.gg (complete VCT season including all 3 regions)

### Success Criteria
- Backtest accuracy against actual VFL prices
- Composition diversity: do the generated prices allow many different viable 100 VP teams?
- Distribution analysis: is the spread healthy or does it force a single dominant archetype?
- Price-to-performance correlation: do prices predict actual fantasy output?

---

## Technical Notes

- All 3 CSVs are standardized to UTF-8 encoding with identical 36 columns (via `standardize_csvs.py`)
- Rows with `P? = 0` and all-zero stats indicate the player did not participate in that game (team eliminated, bye week, etc.)
- The `Game Start VP` and `Game End VP` columns show how VFL's current dynamic pricing moves after each game (2024 scraped data does NOT have VP prices — only 2025/2026 do)
- China teams are in the data but NOT relevant for VFL pricing (filter them out):
  - Bilibili Gaming, Dragon Ranger Gaming, Edward Gaming, FunPlus Phoenix, JD Mall JDG Esports, Nova Esports, Titan Esports Club, Trace Esports, Wolves Esports, Xi Lai Gaming, TYLOO, All Gamers
- 2024 data was scraped from vlr.gg via `scrape_vlr_2024.py` (parallel scraper with 4 workers)
- Kill scoring: 0 kills = -3 pts, 1-4 kills = -1 pts, 5-9 kills = 0 pts, 10+ kills = +1 and up

---

## Algorithms (implemented in `pricing_algorithms.py`)

### 1. Baseline EMA (Exponential Moving Average)
- Compute EMA of PPM with alpha=0.3, 2024 data weighted at 0.5x
- Bayesian shrinkage for players with < 4 games (blend toward population mean)
- Linear map from EMA PPM to [6, 15] VP range
- Simplest approach — serves as the baseline to beat

### 2. Distribution-Aware Pricing
- Same EMA PPM scores, but force a healthy price distribution
- Percentile-rank players, then map to target quantiles: 10th->6.5, 25th->7.5, 50th->9.0, 75th->10.5, 90th->12.5, 95th->13.5
- Fixes the median-below-9 problem and reduces clustering at 8-10 VP
- Goal: enable multiple viable team archetypes

### 3. Role-Adjusted Pricing
- Proxy player roles from scoring patterns (kill ratio, P.Pts/Pts ratio, high kill bracket rate, rating bonus frequency)
- Duelist-like players: discount raw PPM by 5-10% (they naturally score higher from kills)
- Support-like players: boost by 5% (they contribute more than their raw PPM suggests)
- Hook for real role data when pickrate/IGL data arrives

### 4. Team Strength Multiplier
- Decompose expected points into P.Pts (personal) + T.Pts (team-based)
- Predict T.Pts from team win rate (strong teams give ~1.5 extra pts/game)
- Predict P.Pts from personal EMA
- Combine for total expected PPM, then price

### 5. Variance-Aware Pricing
- Start from baseline EMA price
- Adjust by coefficient of variation (std/mean of PPM)
- Consistent players get up to +10% premium (reliable weekly scorers)
- Volatile players get a discount (boom-or-bust)
- Re-clip to [6, 15]

### 6. Combined Pricing (recommended)
- Blends ALL available signals with performance-dominant weighting:
  - 60% EMA performance (core) + role adjustment
  - 12% Team strength (win rate, discounted by roster changes)
  - 8% Pickrate popularity (supply/demand — popular players cost more)
  - 8% Team brand popularity (fan-favorite teams have inflated demand)
  - 7% Opponent strength (facing weaker schedule = slight boost)
  - 5% Consistency premium
- Uses distribution-aware quantile mapping for the final VP assignment
- Closest median to ideal 9.09 (median ~8.97)
- New-player team strength discount: 1 new=0.67x, 2=0.33x, 3+=ignore
- All players tagged with uncertainty (HIGH/MEDIUM/LOW) for manual review

### Pickrate Data (`data/pickrate_data.csv`, `data/pickrate_summary.csv`)
Historical VFL manager pick data from 5 events:
- **Event 1**: 2025 Toronto (Masters 2) — 60 players, 3 gameweeks
- **Event 2**: 2025 Stage 2 — 183 players, 5 gameweeks
- **Event 3**: 2025 Champions — 135 players, 3 gameweeks
- **Event 4**: 2026 Kickoff — 180 players, 1 gameweek
- **Event 5**: 2026 Santiago (Masters 1) — 60 players, 3 gameweeks

Used to compute:
- **Player pickrate**: avg pick % across events (popular players should cost more)
- **Team brand popularity**: avg pick % of all players on a team (fan-favorite bias)
- No IGL rate data in these files (only Name + pickcount)

### Opponent Strength
- In group stages, team matchups are known in advance
- Opponent quality affects expected T.Pts (map wins/losses)
- Currently uses team win rate as a proxy for opponent strength
- Future: could use actual group-stage matchup data for per-gameweek pricing

## Evaluation Framework (`evaluate_pricing.py`)

- **Price accuracy**: MAE, RMSE, correlation vs actual 2025 Stage 1 Game Start VP
- **Composition simulation**: greedy optimizer with 10K random restarts (11 players, ≤100 VP, max 2/team)
- **Archetype diversity**: classify generated teams as top-heavy, balanced, spread, stars-and-scrubs
- **Backtest**: score top 15 teams against actual Stage 1 results, compare vs actual-price teams and random teams

---

# Claude Behaviour Rules

## Spec-Anchored Development

Every new feature or behavior change **must** start with a spec.

### Philosophy

A spec is not documentation about code. A spec is upstream of code — the source of truth for intent. Code implements the spec, not the other way around. When spec and code disagree, the spec needs a conscious decision — not a silent override.

The key difference from "spec-first" (where the spec dies after implementation) is that the spec lives as long as the feature lives, and is updated alongside code for the entire lifetime of that feature.

| Level | Spec lives... | Human edits... |
|-------|---------------|----------------|
| Spec-first | until task is done | code |
| Spec-anchored | while feature lives | spec + code |
| Spec-as-source | forever | only spec, code is generated |

### The 5 Rules

1. **Spec lives in the repo next to code** — not in Confluence, not in a ticket.
2. **Every behavior change = update spec first** — if you change behavior, the spec changes in the same commit/PR.
3. **Spec describes INTENT, not implementation** — what the feature does, why, boundaries, edge cases. Not how (that's the code).
4. **Code review includes spec review** — if a PR changed behavior but the spec wasn't updated, that's a bug in the review process.
5. **AI agents read the spec before making changes** — the agent gets the spec as context, works within those boundaries, and updates the spec when done.

### File Naming

- One file per standalone feature: `feature-name.md`
- Kebab-case: `calendar-preferences.md`, `payment-history.md`
- Short but descriptive

### Standalone vs Epic

- Feature is self-contained, one file covers it → flat spec
- Product initiative has 3+ distinct implementable features sharing motivation → epic directory

### Epic Directory Convention

- `_overview.md` — epic-level context, links to origin PRD, high-level vision. Not a spec itself.
- Child specs — each follows the standard template, self-contained with own status/boundaries/testing/decision log. References `_overview.md` for broader context.

### Spec Template

```markdown
# Feature: [Name]

## Status
draft | approved | implemented | modified | deprecated

## Why this exists
Who asked for it, what pain it solves, business motivation.
Support can read this. Marketing can read this.

## What it does
Behavior description — what the user experiences.
Not code, not implementation. Just: "when X happens, user sees Y."

## Boundaries & edge cases
What it intentionally does NOT do. Known limitations.
This is gold for AI agents — "stay inside these lines."

## Testing & verification
How we know this works. What to automate, what to smoke-test.

### Key scenarios
- [ ] Scenario 1
- [ ] Scenario 2

### Edge cases
- Edge case 1 — expected behavior
- Edge case 2 — expected behavior

### Automation notes
What's worth automating vs manual spot-check.

## Decision log
| Date | Decision | Why | Who |
|------|----------|-----|-----|
| YYYY-MM-DD | What was decided | Reasoning | Who decided |

## Related
- Links to related specs, code paths, or external references
```

### Cross-Cutting Technical Decisions

For decisions that span multiple features or establish reusable patterns:

Naming: `NNN-short-descriptive-name.md` (numbered)

Template:

```markdown
# Decision NNN: Title

## Status
accepted | superseded | deprecated

## Context
What situation or problem prompted this decision.

## Options considered
1. **Option A** — description. Pros/cons.
2. **Option B** — description. Pros/cons.

## Decision
What we chose and why.

## Consequences
What follows — what's easier, what's harder, what to watch for.

## Applies to
Which specs, features, or areas of the codebase this affects.
```

### Rules for AI Agents

When working on a feature with an existing spec:
1. Read the spec first — before touching any code
2. Work within the boundaries — the spec defines what the feature should do
3. Use testing scenarios as acceptance criteria
4. Update the spec when done — if you changed behavior, the spec must reflect it
5. Update the decision log — if you made a non-obvious choice, log it with reasoning

When creating a new feature:
1. Create the spec first — before writing code
2. Get approval on the spec — align on intent before implementation
3. Implement within spec boundaries — the spec is the contract
4. Fill in testing scenarios — as you discover edge cases during implementation

Mandatory highlights:
- **Before writing code** — create or update the spec for the feature being changed.
- **Spec and code change together** — spec updates ship in the same commit/PR as code changes.
- **Spec describes intent, not implementation** — what, why, boundaries, edge cases. Code handles the how.
- **AI agents**: before modifying a feature, check for an existing spec. Read it first. Update it when done. Remind the user if a behavior change doesn't have a corresponding spec update.
- **Cross-cutting technical decisions** should be recorded. Check existing decisions before making architectural choices that span multiple features.

---

## Code Size & Complexity Limits

Small functions and shallow nesting are non-negotiable. If a function is hard to read, it's wrong — split it.

### Function/Method Length

| Language | Max lines per function | Hard ceiling |
|----------|----------------------|--------------|
| Python | 20 lines | 30 lines (requires justification in PR) |
| TypeScript | 20 lines | 30 lines |

Lines = logical lines, not counting blank lines, comments, or type annotations. If you're at 20, look for an extract. If you're at 30, you've already gone too far.

### Nesting Depth

| Language | Max indentation levels | Meaning |
|----------|----------------------|---------|
| Python | 3 levels | function → if → for (stop here) |
| TypeScript | 3 levels | function → if → for (stop here) |

If you need a 4th level, extract the inner block into a named function. Early returns, guard clauses, and `continue` are your friends:

```python
# BAD — 4 levels deep
def process_orders(orders):
    for order in orders:
        if order.is_valid:
            if order.side == "buy":
                if order.price > 0:
                    execute(order)

# GOOD — flat
def process_orders(orders):
    for order in orders:
        if not order.is_valid:
            continue
        if order.side != "buy":
            continue
        if order.price <= 0:
            continue
        execute(order)
```

### File Size

| Language | Target max lines | Hard ceiling |
|----------|-----------------|--------------|
| Python | 200 lines | 300 lines |
| TypeScript/TSX | 200 lines | 300 lines |
| Test files | 400 lines | 600 lines |

If a file hits 200 lines, look for a natural split. If it hits 300, split it before adding more code.

### Enforcement

- **Code review**: reject PRs that violate these limits without justification
- **AI agents**: when writing code, check function length and nesting before finishing. Split proactively — don't wait for review.
- **Exceptions**: data class definitions, configuration blocks, and auto-generated code are exempt from line counts

---

## Test Writing Rules

### Core Philosophy

Based on [Testing Overview](https://abseil.io/resources/swe-book/html/ch12.html) and [Test Doubles](https://abseil.io/resources/swe-book/html/ch13.html).

### Rule 1: Test Behaviors, Not Methods

A test should verify what the code does for the user, not mirror internal structure. One method may need multiple tests (multiple behaviors). Multiple methods may need one test (one behavior spanning them).

```
// BAD — one test per method, testing internals
test("calculateTotal()")
test("applyDiscount()")
test("formatPrice()")

// GOOD — one test per behavior the user cares about
test("order with coupon shows discounted total at checkout")
test("order without coupon shows full price")
test("expired coupon is rejected with clear message")
```

### Rule 2: Test State, Not Interactions

Assert on outcomes (return values, database state, UI output), not on whether specific internal functions were called or in what order.

Exception: interaction testing is acceptable when the side effect IS the behavior (e.g., verifying an external API was called in an integration test).

### Rule 3: DAMP > DRY

Tests should be Descriptive And Meaningful Phrases. Repeating setup code across tests is fine if it makes each test self-contained and readable. A reader should understand a test without jumping to shared helpers.

### Rule 4: No Logic in Tests

No `if`, `else`, loops, ternaries, string concatenation, or conditional assertions in test bodies. If you need branching, write separate tests.

### Rule 5: Given / When / Then Structure

Every test has three clearly separated phases:
1. **Given** — set up the preconditions
2. **When** — perform the single action under test
3. **Then** — assert on outcomes

### Rule 6: Make Tests Complete and Concise

- Include all relevant setup in the test body (complete)
- Only set fields that affect the behavior under test (concise)
- Use factory defaults for everything else

### Rule 7: Test Names Should Be Sentences

A test name should describe the behavior, not the method. Pattern: "[unit] [does something] [when/given condition]".

### Rule 8: Prefer Real Implementations Over Mocks

Use real implementations wherever practical. Only use test doubles when:
- The real implementation is slow (network, disk, heavy computation)
- The real implementation is nondeterministic (time, randomness, external services)
- The real implementation is unavailable in the test environment

When you must mock, prefer fakes > stubs > mocks.

### Rule 9: Don't Overuse Mocks

Signs you're over-mocking:
- Mock setup is longer than the test itself
- You're mocking things you own
- You're asserting on mock call arguments rather than outcomes

### Rule 10: Tests Must Be Deterministic

No dependency on wall-clock time, execution order, external services, random data, or shared mutable state between tests.

### Rule 11: Make Tests Hermetic

Each test sets up everything it needs, runs in isolation, and cleans up after itself. Must produce the same result whether run alone, in a suite, or in parallel.

### Rule 12: Tests Should Be Fast

- Unit tests: < 2 seconds per test
- Never `sleep()` in a test — use polling with short intervals or event-driven waits
- Slow tests should be classified differently (integration) and run in a separate suite

### Rule 13: Unchanging Tests

The ideal test is written once and never changed unless the behavior it covers changes. If refactoring internals breaks your tests, they're testing implementation, not behavior.

### Rule 14: One Assertion Per Behavior (Not Per Test)

Multiple asserts in one test are fine when they verify different facets of one behavior. Multiple asserts verifying different behaviors should be separate tests.

### Rule 15: Test Public APIs, Not Internal Details

Test through the same interface your callers use. If a function is private/internal, test it through the public function that calls it.

### Test Types

| Type | What it tests | Dependencies | Speed |
|------|---------------|--------------|-------|
| Unit | Business logic in isolation | All external deps mocked/faked | < 2s per test |
| Integration | Components working together, real I/O | Real database, file system, etc. | 2-30s per test |
| UI / Component | Rendering, interaction, accessibility | DOM environment, usually mocked backend | Medium |
| End-to-end | Full user workflows through the real system | Everything real | Slow |

Sizing guidance: ~80% unit, ~15% integration, ~5% end-to-end.

### Test Data

1. Use factories — never construct test objects by hand with all fields
2. Override only relevant fields — let factory defaults handle the rest
3. Use deterministic IDs — predictable test data is easier to debug
4. Clean up after tests — use shared cleanup utilities

### Hard Rules

1. Never commit with failing tests — all tests pass before merge
2. No feature is complete without tests
3. Never dismiss failures as "pre-existing" — fix or escalate
4. No arbitrary timeouts — fix root causes of slowness
5. No `eslint-disable` / `lint:ignore` — fix the issue or split the test
6. Keep test files short (under 600 lines) — split into focused files
7. Don't duplicate test utilities — use shared helpers
8. Run fast checks first — lint → typecheck → unit → integration

### References

- https://abseil.io/resources/swe-book/html/ch11.html
- https://abseil.io/resources/swe-book/html/ch12.html
- https://abseil.io/resources/swe-book/html/ch13.html
- https://abseil.io/resources/swe-book/html/ch14.html

---

## Experimentation & Scientific Method

Every investigation is an experiment. Treat it like science, not hacking.

### 1. Before You Start

- Check what's already known. Search existing findings before running anything.
- State a hypothesis. "I think X will happen because Y." If you can't state a hypothesis, you don't understand the problem yet.
- One variable at a time. If you change two things and it works, you don't know which one fixed it.

### 2. Structure

- Every experiment gets its own directory
- Every experiment gets a `README.md` with: hypothesis, setup, procedure, raw results, conclusion
- Every experiment gets an entry in an index with a one-line finding summary
- If it modifies production code, it MUST be on a dedicated branch

### 3. During the Experiment

- Document everything. Exact commands, raw output, timestamps.
- Document failures. They're MORE valuable than successes — they eliminate paths.
- Reproduce before concluding. One success is an anecdote. Three successes is evidence.
- Don't fix mid-experiment. If something breaks, document the failure, then start a new attempt with the fix.

### 4. After the Experiment

- Extract durable findings into a findings document
- Distinguish what you built from what you observed. Code design = spec. Hardware/library behavior = finding. Choice between alternatives = decision.
- Update the index with key finding.
- Don't merge experiment branches until findings are proven correct and code is ready for production.

### 5. Audits (Reviewing Past Work)

- Re-read findings with fresh eyes. Are they still accurate?
- Check if experiment conclusions made it into specs or code. If not, why?
- Flag stale findings — mark them with a date and "needs re-validation" if the hardware/software has changed since.
