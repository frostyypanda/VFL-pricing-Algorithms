# VFL 2026 Stage 1 — GW3 Transfer Plans (T1 & T2)

**Generated:** 2026-04-13
**Data in:** W1 EMEA+PAC (6+6 matches) + W2 EMEA+PAC (6+6) + W1 AMER (6) = **30 matches**
**Model:** EB-shrunk EP + RidgeCV features + 2-week Bayesian update (α=0.25/week)
**Prices used:** `Stats for VFL Prices 2026 — Stage 1 VFL Pricing Sheet.csv` (Final Cost)

---

## Schedule reminder

| GW | Regions playing | Transfer budget |
|---|---|---|
| GW3 | EMEA + PAC + AMER | 3 |
| GW4 | EMEA + PAC + AMER | 3 |
| GW5 | EMEA + PAC + AMER | 3 |
| GW6 | **AMER only** | 3 |

**Hard constraint added:** team must contain ≥11 AMER players by GW6 (non-AMER score 0 that week). Intermediate targets to make the ramp feasible: ≥2 AMER after GW3, ≥5 after GW4, ≥8 after GW5.

---

## T1 — Summary

Starting AMER: **1/11** (Eggsterr) | Total VP: 100.0

| GW | Transfers | Projected Pts (2x IGL) | IGL | AMER |
|---|---|---|---|---|
| GW3 | Lar0k→Primmie, Kushy→Killua, BeYN→Saadhak | **116.2** | Primmie | 2/11 |
| GW4 | Patmen→kingg, Rosé→mwzera, Free1ng→Tex | **113.0** | Jemkin | 5/11 |
| GW5 | Primmie→Alym, Veqaj→Zekken, Yetujey→Cryocells | **111.5** | Zekken | 8/11 |
| GW6 | Killua→Asuna, Jemkin→Lukxo, Udotan→Sato | **112.6** | Zekken | **11/11** |

**Total projected (GW3-6):** **453.3 pts**

### T1 — key reads
- **Protect Jemkin/Primmie through their best weeks.** Jemkin is priced at 13.5 but holds EP through GW5; only sell for GW6 when PAC is off.
- **GW3 IGL: Primmie** (scraped 3-0 W2, strong rating form). Alternative: Jemkin.
- **GW4-6 IGL: Zekken** once on board — best AMER duelist per post-W1 data, 12.1 GW6 EP with 2x = **24.2** pts.
- **Don't** transfer Eggsterr early even though his EP is low (~6.0); his 7.5 VP is locked budget you need for GW6 AMER rebuild.

Full week-by-week table in `GW3_transfer_plan_T1.md`.

---

## T2 — Summary

Starting AMER: **0/11** | Total VP: 100.0 — tougher re-build because nothing to anchor on.

| GW | Transfers | Projected Pts (2x IGL) | IGL | AMER |
|---|---|---|---|---|
| GW3 | xeus→Alym, Kushy→Killua, Yetujey→Tex | **115.9** | Killua | 2/11 |
| GW4 | PROFEK→Saadhak, Rosé→mwzera, BeYN→Cryocells | **111.5** | Alym | 5/11 |
| GW5 | Veqaj→Zekken, Free1ng→Lukxo, Patmen→cauanzin | **114.3** | Zekken | 8/11 |
| GW6 | Derke→Sato, Killua→brawk, Udotan→bang | **113.3** | Zekken | **11/11** |

**Total projected (GW3-6):** **455.0 pts**

### T2 — key reads
- **Don't sell Derke in GW3.** His 14VP is the most expensive slot on the team, but his EP (11.5 GW3, 10.3 GW4, 12.5 GW5) is high enough that rotating him out early costs more than it saves. Plan sells him in GW6 for Sato (LEVIATÁN, 13VP) as AMER duelist.
- **Kushy → Killua (GW3)** is a lateral swap (same 8.0 VP, same Initiator slot) — Killua's post-W2 form shoots him to ~12.8 GW3 EP, making him an IGL candidate.
- **GW3 IGL: Killua** on Full Sense's tactical tear (13-5, 13-7 maps in W2).
- **GW4 IGL swings to Alym** (FURIA's AMER W1 was 2-1 vs NRG, he top-fragged). Then Zekken takes over GW5-6.
- **Double KRÜ risk:** Saadhak + mwzera both land on KRÜ by GW4. Keep an eye on KRÜ schedule — they play Evil Geniuses GW5, a tough matchup.

Full week-by-week table in `GW3_transfer_plan_T2.md`.

---

## Cross-team notes

**Shared core across both teams by GW6:**
- Zekken (MIBR) — IGL choice 3 of 4 gameweeks on T1, 2 of 4 on T2. Pricing sheet has him at 12.5, still worth it.
- Tex (MIBR) — 8.0 VP AMER sentinel, high-floor
- Cryocells (100T) — 10.5 VP wildcard workhorse
- Alym (FURIA) — 10.0 VP, versatile D/W
- mwzera (KRÜ) — 6.5 VP initiator, strong EP-per-VP

**Max-2-per-team** binding constraints to watch:
- T1 GW5-6: MIBR = Zekken + Tex (max). Can't add Zaajin / bang to MIBR etc.
- T2 GW5-6: LOUD = Lukxo + cauanzin (max). KRÜ = Saadhak + mwzera (max).

**IGL cycle (combined, both teams same picks unless noted):**

| GW | T1 IGL | T2 IGL | Rationale |
|---|---|---|---|
| GW3 | Primmie | Killua | Full Sense on form; T1 has Primmie as duelist, T2 has Killua as initiator |
| GW4 | Jemkin | Alym | T1 still has Jemkin (high EP); T2 doesn't (used transfer slot) |
| GW5 | Zekken | Zekken | Best pure AMER duelist on both teams |
| GW6 | Zekken | Zekken | AMER-only week; Zekken caps EP (12.1 → 24.2 with 2x) |

---

## Caveats & assumptions

1. **Prices are the fixed Stage 1 manual prices** (`Stage1_Price` / Final Cost column from the pricing sheet) — constant for the whole split, as per VFL rules.
2. **EP model** blends historical (EB-shrunk avg Pts) with 2-week Bayesian update. Small-sample bias still present for rookies (e.g. brawk, cauanzin) — their EPs are shrunk toward population means.
3. **Amer bonus in objective** (GW3=0.5, GW4=1.5, GW5=3.0) encourages the optimizer to front-load AMER acquisition beyond the hard minimums.
4. **GW6 AMER minimum = 11** is hard-enforced. If you'd rather run GW6 with 10/11 AMER and keep one high-EP PAC player through GW5, drop the min to 10 (easy code change: `targets[6] = 10`).
5. **IGL choice** is whichever player in the optimal lineup has the highest projected EP for that gameweek — NOT accounting for captaincy risk (benching, bye week mid-tournament, etc.). Spot-check roster news before locking.

---

## Files

- `GW3_transfer_plan_T1.md` — full week-by-week table for Team 1
- `GW3_transfer_plan_T2.md` — full week-by-week table for Team 2
- `data/w1_vlr_results.json` — scraped W1 EMEA+PAC (12 matches)
- `data/w2_vlr_results.json` — scraped W2 EMEA+PAC + W1 AMER (18 matches)
- `generate_gw3_report.py` — reproducible planner (python generate_gw3_report.py)
- `scrape_gw2.py` — reproducible scraper for GW2 matches
