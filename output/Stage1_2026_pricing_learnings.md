# Stage 1 2026 — Pricing Model Eval & Learnings

**Date:** 2026-05-13
**Backtest scope:** Predict 2026 Stage 1 prices from pre-Stage 1 data only (2024 full + 2025 full + 2026 Kickoff + 2026 Santiago), then score predictions against actual Stage 1 performance.
**Models:** Manual (committee Stage1_Price) vs Algo1 (archive `algo_combined`) vs Algo2 (v2 EB+Ridge+EMA ensemble + quantile mapping).

---

## 1. Headline result

| Model | Pearson r | Spearman ρ | MAE (price vs PPG) | Mean | Median | Unique prices | Top-20 hit | Rank MAE |
|---|---|---|---|---|---|---|---|---|
| **Manual** | **0.572** | **0.530** | 2.63 | 9.25 | 8.5 | 21 | **9/20** | **34.8** |
| Algo1 (Combined) | 0.343 | 0.341 | 3.50 | 10.21 | 10.0 | 19 | 7/20 | 42.2 |
| Algo2 (v2) | 0.389 | 0.372 | 2.77 | 9.11 | 8.5 | 20 | 7/20 | 40.6 |

**Manual outperforms both algorithms on every overall metric.** This is *not* the criterion we optimize for (per project memory — old price accuracy is not the goal), but it does mean the algorithms are not yet beating the human committee, which is what they need to do to be worth using.

Between the two algorithms, **Algo2 beats Algo1** on every metric except slightly fewer top-N hits — even though Algo1's "Combined" pipeline has more hand-engineered signals (pickrate, team popularity, opponent strength), they're noisy and don't pay off. The simpler v2 ensemble (Empirical Bayes + Ridge + EMA + quantile mapping) is more disciplined.

---

## 2. Factor breakdown (Pearson r within cohort)

| Cohort | Manual | Algo1 | Algo2 | n |
|---|---|---|---|---|
| Rookies (0 prior games) | n=1 | n=1 | n=1 | 1 |
| Green (1-10 prior games) | +0.346 | +0.181 | **+0.049** | 48 |
| Veterans (30+ prior games) | +0.620 | +0.314 | +0.453 | 57 |
| AMER | +0.374 | **+0.039** | +0.147 | 50 |
| EMEA | +0.598 | +0.403 | +0.311 | 55 |
| PAC | +0.704 | +0.483 | +0.604 | 54 |

**Observations:**
1. **All models price PAC players best** — PAC had the most stable rosters and the most pre-Stage 1 played games to learn from.
2. **AMER is brutal for the algorithms.** Algo1 hits r=0.039 (near zero) on AMER players. Algo2 isn't much better at r=0.147. Manual's r=0.374 is also its weakest cohort. **Correction (2026-05-14):** an earlier version of this doc said AMER lacked Kickoff/Santiago data — that was wrong. AMER has 239 Kickoff played rows + 45 Santiago rows. Mean pre-Stage-1 game count for AMER players is 24.2 — comparable to EMEA (24.9) and PAC (27.4). The real driver is **roster churn**: 4 AMER teams (ENVY, Evil Geniuses, FURIA, G2 Esports) have 100% new starting fives vs their 2025 squads. The historical data exists for those players, but they're now on different teams playing different roles/comps, so team-context features mislead.
3. **Algo2 collapses on green players (r=0.049).** Empirical Bayes shrinks them too hard toward the population mean, washing out the predictive signal that's actually there. This is the single most actionable failure mode.
4. **Veterans is where Algo2 closes the gap to Manual** (0.453 vs 0.620) — when prior data is plentiful, the EB+Ridge+EMA ensemble is competitive.

---

## 3. Where everyone whiffed: roster change & form drift

These players were badly overrated by **all three** models — all three had strong prior data and the algos (and humans) didn't reweight for form/roster changes that came in 2026 Stage 1:

| Player | Team | Actual PPG | Manual | Algo1 | Algo2 | Actual rank |
|---|---|---|---|---|---|---|
| JohnQT | Sentinels | 5.0 | 9.0 | **15.0** | 9.5 | 124 |
| Lovers rock | BBL Esports | 5.2 | 13.0 | 9.5 | **15.0** | 119 |
| skuba | NRG Esports | 5.4 | 9.5 | 14.0 | 13.5 | 116 |
| Lar0k | BBL Esports | 5.8 | 12.0 | 10.5 | 13.5 | 105 |
| Loita | BBL Esports | 4.0 | 8.5 | 9.5 | 11.0 | 146 |
| Avez | Karmine Corp | 3.2 | 8.0 | 11.5 | 9.5 | 155 |
| ZynX | Gen.G | 3.0 | 9.0 | 11.0 | 8.5 | 158 |
| Xeppaa | Cloud9 | 4.6 | 7.5 | 12.0 | 8.0 | 135 |

**Three of the worst overrates are BBL Esports players.** BBL had a strong 2025 (high prior PPM), and every model relied on that. None of the models have a mechanism to detect that BBL the *team* dropped a level between 2025 and 2026 Stage 1. Same story for Karmine Corp.

And these were badly **underrated** by all three:

| Player | Team | Actual PPG | Manual | Algo1 | Algo2 | Actual rank |
|---|---|---|---|---|---|---|
| Favian | Eternal Fire | 9.8 | 7.5 | 7.5 | **5.5** | 16 |
| Lukxo | LOUD | 10.0 | 9.5 | 9.0 | 9.0 | 14 |
| Saadhak | KRÜ | 9.2 | 6.0 | 9.0 | 7.0 | 24 |
| oonzmlp | VARREL | 9.2 | 6.5 | 6.5 | 6.0 | 24 |
| mwzera | KRÜ | 8.4 | 6.5 | 6.0 | **5.5** | 43 |
| Darker | LOUD | 8.4 | 6.5 | 7.5 | **5.5** | 43 |
| Reduxx | Sentinels | 9.0 | 11.0 | 12.0 | 6.5 | 30 |
| Spike | LEVIATÁN | 8.2 | 8.0 | 8.5 | 6.0 | 48 |

Common thread: mid-tier players (mostly EMEA/AMER) who broke out in 2026 Stage 1. The algorithms had no way to anticipate elevation, and even Manual mostly missed it.

**Two specific observations** the team and I should remember:
- **Reduxx's underrate by Algo2 (6.5)** is striking — Manual had her at 11.0 and Algo1 at 12.0. Algo2's quantile mapping put her in the bottom third because her pre-Stage 1 sample size was small and EB pulled her hard to the mean.
- **Saadhak** moved to KRÜ — Manual *did* spot this and price him low (6.0) on the assumption that the team would struggle. He actually performed well; the team change was the wrong direction to lean.

---

## 4. Why Algo1 (Combined) underperforms Algo2 (v2)

Algo1 has a richer feature mix (60% EMA + team strength + pickrate + brand + opponent + consistency), but performs worse than Algo2's leaner pipeline. Reading the code, the issues:

- **Inflated price distribution.** Algo1's mean price is 10.21 (target is ~9.09). Its quantile anchors push mid-tier prices up. This makes its MAE worse (3.50 vs 2.77) even when its rank correlation is similar.
- **Pickrate signal is noise here.** F5 in the spec said pickrate→performance r=0.11, yet Algo1 gives it 8% weight. The eval suggests this 8% is hurting more than helping — pickrate is reflecting last season's hype, not 2026 form.
- **Team brand popularity (8%) is correlated with prior form**, so it's effectively double-counting team strength. BBL and Karmine Corp had high brand popularity going into Stage 1 — exactly the teams that overrate the most.
- **Opponent strength proxy (7%)** uses team WR as an inverse opponent quality proxy, which is circular and shallow.

Algo2 sidesteps these by using only EB + Ridge + EMA blended via average, then a calibrated quantile map. The downside (shown in the green-player cohort) is over-shrinkage.

---

## 5. Recommendations for updating Algo 2

Listed in order of expected impact. None of these are implemented yet — they're hypotheses to test next.

### 5.1 Reduce shrinkage for low-sample players (high impact for "green" cohort, r=0.05)

The EB shrinkage formula `B = τ² / (τ² + σ²/n)` collapses to almost B=0 (full shrinkage to μ) for n in 1-10. For these players, the per-player signal is *weak but not zero*, and right now it's effectively discarded.

Try: a min-B floor (e.g., B≥0.3 even for n=3) so the player's own data retains weight. Or use a two-stage approach: shrink toward role-mean rather than population mean (a 5-game duelist sample should shrink toward the duelist mean, not the league mean).

### 5.2 Add team-form decay term (high impact for the BBL/Karmine overrate pattern)

Define team form recency: weight each team's most recent stage results more heavily when computing team_wr. Then have a player's expected points discount by `(team_form_now / team_form_historical)`.

Concrete: BBL's 2025 stage WR was high, their Santiago WR was middling. A decay-weighted team factor would have caught this. The current calibration uses pooled WR across all training data, missing the trend.

### 5.3 Replace pickrate signal with role-mean shrinkage

Algo1's pickrate component is a dead weight (it correlates with last-season hype, not 2026 performance). Algo2 doesn't use it currently and we shouldn't be tempted to add it. Where Algo2 could borrow from Algo1: the **kill_profile + ppts_ratio role proxies**, which let a duelist's high-variance kill scoring be priced differently from a sentinel's steady team-pts contribution.

### 5.4 Cap upside on stars with declining usage

For players priced >12 VP, require *both* high historical PPM **and** consistent recent (≤1 stage prior) performance. JohnQT and skuba both had strong 2024-2025 baselines but soft Santiago/Kickoff signals — a simple "recent-vs-historical-ratio < 0.85 → cap price at 11.0" guardrail would have caught both.

### 5.5 Roster-stability / team-change feature (root cause of AMER cohort failure)

AMER's lower r is driven by roster churn, not data thinness (verified: AMER has comparable pre-Stage 1 data depth). The actionable signal:

- For each player, compute `team_continuity` = fraction of their prior games played on their *current* team.
- When `team_continuity < 0.3` (player moved teams recently), reduce confidence in their historical PPM (shrink harder toward role+region prior, less toward their personal mean).
- Separately calibrate the quantile mapping for AMER vs the global pool, since AMER's PPM distribution shape may differ.

### 5.6 What NOT to add

- **Pickrate weighting.** Confirmed dead weight in this backtest.
- **Brand popularity.** Same — circular with team strength.
- **Variance/consistency premium.** F3 in spec was already cautious; this eval gives no signal to revisit.

---

## 6. Sidebar: the auto-generated user team finished 3000/6500

User reports the team I generated and transferred each week finished ~3000 / 6500 — middle of the pack, not strong. Consistent with the eval result: the algorithms' expected-points were not better than the manual prices the league was already using.

That said, the transfer plan also had the AMER-11-by-GW6 constraint forcing late-season AMER pickups, which the EP model couldn't optimize cleanly. Worth separating the "pricing quality" finding from the "transfer planner" finding in any follow-up.

---

## 7. Files written this run

- `data/2026 VFL.csv` — updated: 829 played Stage 1 rows (W1–W5 across all regions), playoff placeholders kept as P?=0.
- `data/w6_vlr_results.json` — newly scraped AMER W5 matches (= GW6).
- `output/Stage1_2026_pricing_eval_summary.json` — overall metrics + factor breakdown.
- `output/Stage1_2026_per_player_prices.csv` — Manual / Algo1 / Algo2 prices per player + actual PPG.
- `data/2026 VFL.csv.backup_premerge` — pre-merge backup of the CSV.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-05-13 | VP cols in new Stage 1 rows = manual price (constant) | We don't have the VFL dynamic-pricing formula. Eval doesn't depend on these. |
| 2026-05-13 | Playoff placeholder rows kept as P?=0 | Stage 1 playoffs not scraped yet. |
| 2026-05-13 | Players with 5-9 kills not bucketed in any kill column | Verified empirically against 2025 CSV: bracket math matches P.Pts 100% only under this interpretation. |
| 2026-05-13 | Algo1 = archive `algo_combined`, Algo2 = v2 pipeline | User-confirmed comparison choice. |
