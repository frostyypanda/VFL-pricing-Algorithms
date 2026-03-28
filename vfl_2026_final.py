"""
VFL 2026 Stage 1 — Final Pricing & Team Recommendations.

Runs the v3 pricing algorithm, builds optimal team recommendations
for Manual and Generated pricing schemes, creates an Excel workbook,
and generates the pricing documentation.

Usage:
    python vfl_2026_final.py
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
import copy
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = os.path.dirname(os.path.abspath(__file__))

from pricing_algorithms import (
    compute_team_win_rates, load_pickrate_data,
    compute_player_pickrates, compute_team_popularity,
    CHINA_TEAMS,
)
from schedule_2026 import (
    SCHEDULE, GROUPS, GW_REGIONS,
    get_team_opponent, get_playing_teams,
)
from vfl_2026_v3 import (
    load_all_data_3years, load_manual_prices, load_schedule_opponent_strength,
    generate_v3_prices, _normalize_team_name,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
VP_MIN, VP_MAX = 5.0, 15.0
BUDGET = 100
SQUAD_SIZE = 11
TARGET_MEAN = 100.0 / 11  # ~9.09
NUM_GWS = 6
MAX_TRANSFERS = 3
RNG = np.random.default_rng(2026)

ROLE_SLOTS = {"D": 2, "C": 2, "I": 2, "S": 2}
WILDCARD_SLOTS = 3


# ============================================================================
#  TEAM RECOMMENDATION ENGINE
# ============================================================================

def get_team_region_normalized(team):
    norm = _normalize_team_name(team)
    for region, groups in GROUPS.items():
        for gname, tlist in groups.items():
            for t in tlist:
                if _normalize_team_name(t) == norm or t == team or t == norm:
                    return region
    return None


def compute_player_avg_ppm(training_df):
    result = {}
    for player, grp in training_df.groupby("Player"):
        ppm_vals = grp["PPM"].values
        year_vals = grp["Year"].values
        weights = np.where(year_vals == 2024, 0.5,
                  np.where(year_vals == 2025, 1.0, 2.0))
        if weights.sum() > 0:
            result[player] = np.average(ppm_vals, weights=weights)
        else:
            result[player] = np.mean(ppm_vals)
    return result


def get_player_expected_pts_gw(player, team, gw, player_avg_ppm, team_wr):
    playing_teams = get_playing_teams(gw)
    norm_playing = {_normalize_team_name(t) for t in playing_teams}
    norm_team = _normalize_team_name(team)

    if norm_team not in norm_playing:
        return 0.0

    base_ppm = player_avg_ppm.get(player, 3.0)
    avg_maps = 2.3

    opp = None
    for t_raw in playing_teams:
        if _normalize_team_name(t_raw) == norm_team:
            opp_raw = get_team_opponent(t_raw, gw)
            if opp_raw:
                opp = _normalize_team_name(opp_raw)
            break

    opp_mult = 1.0
    if opp:
        opp_wr = team_wr.get(opp, 0.5)
        if opp_wr > 0.6:
            opp_mult = 0.85
        elif opp_wr < 0.4:
            opp_mult = 1.15

    return base_ppm * avg_maps * opp_mult


def _check_role_validity(squad):
    role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
    for p in squad:
        role = p.get("role", "I")
        if role in role_counts:
            role_counts[role] += 1
    return all(role_counts[r] >= 2 for r in ROLE_SLOTS)


def _build_initial_team(players, gw, budget, n_iter):
    best_team = None
    best_score = -1

    for iteration in range(n_iter):
        noise = RNG.normal(0, 0.3, len(players))
        scored = [(p, p["gw_pts"] / (p["price"] + 0.1) + noise[i])
                  for i, p in enumerate(players)]
        scored.sort(key=lambda x: x[1], reverse=True)

        team = []
        team_vp = 0.0
        team_counts = {}
        role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
        wildcards_used = 0

        for p, _ in scored:
            if len(team) >= SQUAD_SIZE:
                break
            vp = p["price"]
            t = p["Team"]
            role = p["role"]

            if team_vp + vp > budget:
                continue
            if team_counts.get(t, 0) >= 2:
                continue
            remaining = SQUAD_SIZE - len(team) - 1
            if remaining > 0 and (budget - team_vp - vp) < remaining * VP_MIN:
                continue

            if role_counts.get(role, 0) < ROLE_SLOTS.get(role, 0):
                role_counts[role] += 1
            elif wildcards_used < WILDCARD_SLOTS:
                wildcards_used += 1
            else:
                continue

            team.append(p)
            team_vp += vp
            team_counts[t] = team_counts.get(t, 0) + 1

        if len(team) == SQUAD_SIZE and _check_role_validity(team):
            total_pts = sum(p["gw_pts"] for p in team)
            best_player = max(team, key=lambda p: p["gw_pts"])
            total_with_igl = total_pts + best_player["gw_pts"]

            if total_with_igl > best_score:
                best_score = total_with_igl
                best_team = {
                    "players": list(team),
                    "total_vp": round(team_vp, 2),
                    "expected_pts": round(total_pts, 1),
                    "expected_pts_with_igl": round(total_with_igl, 1),
                    "igl": best_player["Player"],
                    "transfers": [],
                }
    return best_team


def _optimize_transfers(all_players, current_squad, gw, budget,
                        max_transfers, n_iter, amer_bonus=0.0):
    current_names = {p["Player"] for p in current_squad["players"]}
    current_vp = current_squad["total_vp"]
    player_lookup = {p["Player"]: p for p in all_players}

    for p in current_squad["players"]:
        if p["Player"] in player_lookup:
            p["gw_pts"] = player_lookup[p["Player"]]["gw_pts"]
        else:
            p["gw_pts"] = 0

    best_result = None
    best_score = -1

    for iteration in range(n_iter):
        squad = [dict(p) for p in current_squad["players"]]
        squad_names = set(current_names)
        squad_vp = current_vp
        transfers_made = []

        n_transfers = int(RNG.integers(0, max_transfers + 1))

        for _ in range(n_transfers):
            squad_pts = [(i, p["gw_pts"]) for i, p in enumerate(squad)]
            squad_pts.sort(key=lambda x: x[1] + float(RNG.normal(0, 2)))
            out_idx = squad_pts[0][0]
            out_player = squad[out_idx]

            candidates = [p for p in all_players
                         if p["Player"] not in squad_names
                         and p["price"] <= budget - squad_vp + out_player["price"]]

            if not candidates:
                continue

            noise = RNG.normal(0, 1.0, len(candidates))
            scored_candidates = []
            for i, c in enumerate(candidates):
                bonus = amer_bonus if c.get("region") == "AMER" else 0
                scored_candidates.append((c, c["gw_pts"] + bonus + float(noise[i])))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            for cand, _ in scored_candidates[:5]:
                new_vp = squad_vp - out_player["price"] + cand["price"]
                if new_vp > budget:
                    continue
                team_count = sum(1 for p in squad if p["Team"] == cand["Team"]
                               and p["Player"] != out_player["Player"])
                if team_count >= 2:
                    continue

                test_squad = [p for p in squad if p["Player"] != out_player["Player"]]
                test_squad.append(cand)

                role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
                wildcards = 0
                valid = True
                for tp in test_squad:
                    role = tp["role"]
                    if role_counts.get(role, 0) < ROLE_SLOTS.get(role, 0):
                        role_counts[role] += 1
                    elif wildcards < WILDCARD_SLOTS:
                        wildcards += 1
                    else:
                        valid = False
                        break
                if not valid:
                    continue

                squad = test_squad
                squad_names = {p["Player"] for p in squad}
                squad_vp = new_vp
                transfers_made.append((out_player["Player"], cand["Player"]))
                break

        total_pts = sum(p["gw_pts"] for p in squad)
        if squad:
            best_player = max(squad, key=lambda p: p["gw_pts"])
            total_with_igl = total_pts + best_player["gw_pts"]
        else:
            total_with_igl = 0
            best_player = {"Player": "N/A"}

        if total_with_igl > best_score:
            best_score = total_with_igl
            best_result = {
                "players": [dict(p) for p in squad],
                "total_vp": round(squad_vp, 2),
                "expected_pts": round(total_pts, 1),
                "expected_pts_with_igl": round(total_with_igl, 1),
                "igl": best_player["Player"],
                "transfers": list(transfers_made),
            }

    return best_result


def run_gw_recommendations(prices_df, price_col, player_avg_ppm, team_wr,
                           role_map, label=""):
    print(f"\n  --- {label} GW Recommendations ---")

    all_players = []
    for _, row in prices_df.iterrows():
        region = get_team_region_normalized(row["Team"])
        player_name = row["Player"]
        all_players.append({
            "Player": player_name,
            "Team": row["Team"],
            "price": row[price_col],
            "role": role_map.get(player_name.lower(), row.get("Position", "I")),
            "region": region,
            "gw_pts": 0,
        })

    gw_results = []
    current_squad = None

    for gw in range(1, 7):
        for p in all_players:
            p["gw_pts"] = get_player_expected_pts_gw(
                p["Player"], p["Team"], gw, player_avg_ppm, team_wr
            )

        if gw == 1:
            available = [p for p in all_players
                        if get_team_region_normalized(p["Team"]) in ["EMEA", "PAC"]]
            result = _build_initial_team(available, gw, BUDGET, 5000)
        elif gw == 6:
            result = _optimize_transfers(
                all_players, current_squad, gw, BUDGET, MAX_TRANSFERS, 8000,
                amer_bonus=20.0
            )
        else:
            amer_bonuses = {2: 1.0, 3: 3.0, 4: 5.0, 5: 8.0}
            result = _optimize_transfers(
                all_players, current_squad, gw, BUDGET, MAX_TRANSFERS, 5000,
                amer_bonus=amer_bonuses.get(gw, 1.0)
            )

        if result is None:
            print(f"  GW{gw}: FAILED to find valid team!")
            gw_results.append(None)
            continue

        print(f"\n  GW{gw} ({', '.join(GW_REGIONS[gw])}):")
        if result["transfers"]:
            for out_p, in_p in result["transfers"]:
                print(f"    Transfer: {out_p} -> {in_p}")

        print(f"    {'Player':<18} {'Team':<22} {'VP':>6} {'Role':>4} {'Exp':>7} {'IGL':>4}")
        sorted_p = sorted(result["players"], key=lambda p: p["gw_pts"], reverse=True)
        for p in sorted_p:
            igl = "*" if p["Player"] == result["igl"] else ""
            print(f"    {p['Player']:<18} {p['Team']:<22} {p['price']:>6.1f} "
                  f"{p['role']:>4} {p['gw_pts']:>7.1f} {igl:>4}")
        print(f"    Total VP: {result['total_vp']:.1f}  |  "
              f"Exp: {result['expected_pts']:.1f}  |  "
              f"With IGL: {result['expected_pts_with_igl']:.1f}")

        amer_count = sum(1 for p in result["players"]
                        if get_team_region_normalized(p["Team"]) == "AMER")
        print(f"    AMER players: {amer_count}/11")

        gw_results.append(result)
        current_squad = copy.deepcopy(result)

    return gw_results


# ============================================================================
#  EXCEL GENERATION
# ============================================================================

def create_excel(manual_df, v3_df, manual_gw, generated_gw, role_map):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    # ---- Sheet 1: Team Recommendations ----
    ws1 = wb.active
    ws1.title = "Team Recommendations"

    header_font = Font(bold=True, size=12, color="FFFFFF")
    header_fill_manual = PatternFill(start_color="2E7D32", end_color="2E7D32", fill_type="solid")
    header_fill_generated = PatternFill(start_color="BF360C", end_color="BF360C", fill_type="solid")
    subheader_font = Font(bold=True, size=10)
    subheader_fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
    gw_font = Font(bold=True, size=11, color="000000")
    gw_fill = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")
    igl_fill = PatternFill(start_color="FFFDE7", end_color="FFFDE7", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    section_cols = ["Player", "Team", "VP", "Role", "Exp Pts", "IGL"]
    section_width = len(section_cols)
    sections = [
        ("MANUAL PRICING", header_fill_manual, manual_gw),
        ("GENERATED (ALGO)", header_fill_generated, generated_gw),
    ]

    row = 1
    for sec_idx, (sec_name, sec_fill, _) in enumerate(sections):
        start_col = sec_idx * (section_width + 1) + 1
        cell = ws1.cell(row=row, column=start_col, value=sec_name)
        cell.font = header_font
        cell.fill = sec_fill
        cell.alignment = Alignment(horizontal="center")
        end_col = start_col + section_width - 1
        ws1.merge_cells(start_row=row, start_column=start_col,
                       end_row=row, end_column=end_col)

    row = 2
    for sec_idx in range(2):
        start_col = sec_idx * (section_width + 1) + 1
        for col_offset, col_name in enumerate(section_cols):
            cell = ws1.cell(row=row, column=start_col + col_offset, value=col_name)
            cell.font = subheader_font
            cell.fill = subheader_fill
            cell.border = thin_border

    row = 3
    for gw_idx in range(6):
        gw_num = gw_idx + 1
        for sec_idx, (_, _, gw_data) in enumerate(sections):
            start_col = sec_idx * (section_width + 1) + 1
            cell = ws1.cell(row=row, column=start_col,
                           value=f"GW{gw_num} ({', '.join(GW_REGIONS[gw_num])})")
            cell.font = gw_font
            cell.fill = gw_fill
            ws1.merge_cells(start_row=row, start_column=start_col,
                           end_row=row, end_column=start_col + section_width - 1)
        row += 1

        for sec_idx, (_, _, gw_data) in enumerate(sections):
            start_col = sec_idx * (section_width + 1) + 1
            result = gw_data[gw_idx] if gw_data and gw_idx < len(gw_data) else None
            if result:
                if result.get("transfers"):
                    out_names = [t[0] for t in result["transfers"]]
                    in_names = [t[1] for t in result["transfers"]]
                    transfers_str = f"OUT {', '.join(out_names)} -> IN {', '.join(in_names)}"
                else:
                    transfers_str = "Initial squad" if gw_idx == 0 else "No transfers"
                cell = ws1.cell(row=row, column=start_col, value=f"Transfers: {transfers_str}")
                cell.font = Font(italic=True, size=9)
                ws1.merge_cells(start_row=row, start_column=start_col,
                               end_row=row, end_column=start_col + section_width - 1)
        row += 1

        for player_idx in range(SQUAD_SIZE):
            for sec_idx, (_, _, gw_data) in enumerate(sections):
                start_col = sec_idx * (section_width + 1) + 1
                result = gw_data[gw_idx] if gw_data and gw_idx < len(gw_data) else None
                if result and player_idx < len(result["players"]):
                    sorted_players = sorted(result["players"],
                                           key=lambda p: p["gw_pts"], reverse=True)
                    p = sorted_players[player_idx]
                    is_igl = p["Player"] == result["igl"]

                    ws1.cell(row=row, column=start_col, value=p["Player"]).border = thin_border
                    ws1.cell(row=row, column=start_col + 1, value=p["Team"]).border = thin_border
                    ws1.cell(row=row, column=start_col + 2, value=p["price"]).border = thin_border
                    ws1.cell(row=row, column=start_col + 3, value=p["role"]).border = thin_border
                    ws1.cell(row=row, column=start_col + 4,
                            value=round(p["gw_pts"], 1)).border = thin_border
                    igl_cell = ws1.cell(row=row, column=start_col + 5,
                                       value="IGL" if is_igl else "")
                    igl_cell.border = thin_border
                    if is_igl:
                        igl_cell.font = Font(bold=True, color="FF0000")
                        igl_cell.fill = igl_fill
            row += 1

        for sec_idx, (_, sec_fill, gw_data) in enumerate(sections):
            start_col = sec_idx * (section_width + 1) + 1
            result = gw_data[gw_idx] if gw_data and gw_idx < len(gw_data) else None
            if result:
                cell = ws1.cell(row=row, column=start_col, value="TOTAL")
                cell.font = Font(bold=True)
                ws1.cell(row=row, column=start_col + 2,
                        value=result["total_vp"]).font = Font(bold=True)
                ws1.cell(row=row, column=start_col + 4,
                        value=result["expected_pts"]).font = Font(bold=True)
                ws1.cell(row=row, column=start_col + 5,
                        value=f"IGL: {result['expected_pts_with_igl']:.1f}").font = Font(italic=True, size=9)
        row += 2

    for sec_idx in range(2):
        base = sec_idx * (section_width + 1) + 1
        ws1.column_dimensions[get_column_letter(base)].width = 18
        ws1.column_dimensions[get_column_letter(base + 1)].width = 22
        ws1.column_dimensions[get_column_letter(base + 2)].width = 7
        ws1.column_dimensions[get_column_letter(base + 3)].width = 6
        ws1.column_dimensions[get_column_letter(base + 4)].width = 9
        ws1.column_dimensions[get_column_letter(base + 5)].width = 12
        if sec_idx < 1:
            ws1.column_dimensions[get_column_letter(base + 6)].width = 2

    # ---- Sheet 2: Prices ----
    ws2 = wb.create_sheet("Prices")

    price_headers = ["Player", "Team", "Region", "Role", "Position",
                     "Manual VP", "Generated VP", "Diff", "Uncertainty"]

    for col_idx, header in enumerate(price_headers, 1):
        cell = ws2.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color="37474F", end_color="37474F", fill_type="solid")
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center")

    gen_price_map = {}
    gen_unc_map = {}
    for _, row_data in v3_df.iterrows():
        gen_price_map[row_data["Player"]] = row_data["generated_vp"]
        unc = row_data.get("uncertainty", "")
        gen_unc_map[row_data["Player"]] = unc if isinstance(unc, str) and pd.notna(unc) else ""

    sorted_players = sorted(manual_df["Player"].tolist(),
                           key=lambda p: gen_price_map.get(p, 0), reverse=True)

    light_red = PatternFill(start_color="FFCDD2", end_color="FFCDD2", fill_type="solid")
    light_green = PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid")

    for data_row_idx, player in enumerate(sorted_players, 2):
        mrow = manual_df[manual_df["Player"] == player].iloc[0]
        manual_vp = mrow["Price"]
        gen_vp = gen_price_map.get(player, manual_vp)
        diff = gen_vp - manual_vp
        unc = gen_unc_map.get(player, "")

        ws2.cell(row=data_row_idx, column=1, value=player).border = thin_border
        ws2.cell(row=data_row_idx, column=2, value=mrow["Team"]).border = thin_border
        ws2.cell(row=data_row_idx, column=3,
                value=mrow.get("Region", "")).border = thin_border
        ws2.cell(row=data_row_idx, column=4,
                value=mrow.get("Role", "")).border = thin_border
        ws2.cell(row=data_row_idx, column=5,
                value=mrow.get("Position", "")).border = thin_border
        c6 = ws2.cell(row=data_row_idx, column=6, value=manual_vp)
        c6.border = thin_border
        c7 = ws2.cell(row=data_row_idx, column=7, value=gen_vp)
        c7.border = thin_border
        c8 = ws2.cell(row=data_row_idx, column=8, value=round(diff, 1))
        c8.border = thin_border
        if abs(diff) > 2:
            c8.fill = light_red
        elif abs(diff) <= 1:
            c8.fill = light_green
        c9 = ws2.cell(row=data_row_idx, column=9, value=unc)
        c9.border = thin_border

    widths = [18, 22, 8, 18, 8, 10, 12, 8, 30]
    for i, w in enumerate(widths, 1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    output_path = os.path.join(DIR, "VFL_2026_Stage1_Pricing.xlsx")
    wb.save(output_path)
    print(f"\n  Excel saved to: {output_path}")
    return output_path


# ============================================================================
#  DOCUMENTATION
# ============================================================================

def create_documentation(manual_df, v3_df):
    gen_map = dict(zip(v3_df["Player"], v3_df["generated_vp"]))

    unc_map = {}
    for _, row in v3_df.iterrows():
        unc = row.get("uncertainty", "")
        if isinstance(unc, str) and "HIGH" in str(unc):
            unc_map[row["Player"]] = "HIGH"
        elif isinstance(unc, str) and "MEDIUM" in str(unc):
            unc_map[row["Player"]] = "MEDIUM"

    merged = v3_df.merge(manual_df[["Player", "Price"]], on="Player")
    merged["diff"] = merged["generated_vp"] - merged["Price"]
    merged["abs_diff"] = merged["diff"].abs()
    mae = merged["abs_diff"].mean()
    corr = merged["generated_vp"].corr(merged["Price"])
    within_1 = (merged["abs_diff"] <= 1.0).sum()
    within_2 = (merged["abs_diff"] <= 2.0).sum()
    total = len(merged)

    all_sorted = sorted(manual_df["Player"].tolist(),
                        key=lambda p: gen_map.get(p, 0), reverse=True)
    table_rows = ""
    for p in all_sorted[:30]:
        mrow = manual_df[manual_df["Player"] == p].iloc[0]
        man = mrow["Price"]
        gen = gen_map.get(p, man)
        diff = gen - man
        sign = "+" if diff >= 0 else ""
        table_rows += f"| {p} | {mrow['Team']} | {mrow.get('Position','')} | {man:.1f} | {gen:.1f} | {sign}{diff:.1f} |\n"

    high_rows = ""
    med_rows = ""
    for p, level in sorted(unc_map.items()):
        gen_vp = gen_map.get(p, "N/A")
        team = ""
        match = v3_df[v3_df["Player"] == p]
        if len(match) > 0:
            team = match.iloc[0]["Team"]
        games = match.iloc[0]["games_total"] if len(match) > 0 else 0
        row_str = f"| {p} | {team} | {gen_vp} | {games} |\n"
        if level == "HIGH":
            high_rows += row_str
        else:
            med_rows += row_str

    gen_mean = v3_df["generated_vp"].mean()
    gen_median = v3_df["generated_vp"].median()

    doc = f"""# VFL 2026 Stage 1 — Pricing Algorithm

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
| Correlation | {corr:.3f} |
| MAE | {mae:.2f} VP |
| Within 1.0 VP | {within_1}/{total} ({100*within_1/total:.0f}%) |
| Within 2.0 VP | {within_2}/{total} ({100*within_2/total:.0f}%) |
| Generated mean | {gen_mean:.2f} VP (target: {TARGET_MEAN:.2f}) |
| Generated median | {gen_median:.2f} VP |
| Manual mean | {manual_df['Price'].mean():.2f} VP |

## 4. Price Comparison (Top 30 by Generated VP)

| Player | Team | Pos | Manual | Generated | Diff |
|---|---|---|---|---|---|
{table_rows}
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
{high_rows if high_rows else "| (none) | | | |\\n"}
**MEDIUM Uncertainty**:

| Player | Team | Generated VP | Games |
|---|---|---|---|
{med_rows if med_rows else "| (none) | | | |\\n"}
---

*Generated {datetime.now().strftime('%Y-%m-%d')} by VFL Pricing Algorithm v3.1*
"""

    doc_path = os.path.join(DIR, "VFL_2026_PRICING_DOCUMENT.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\n  Documentation saved to: {doc_path}")
    return doc_path


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("VFL 2026 Stage 1 — Final Pricing & Team Recommendations")
    print("=" * 70)

    # --- Step 1: Load data ---
    print("\n[1/4] Loading data...")
    all_data = load_all_data_3years()
    manual_df = load_manual_prices()
    pickrate_df = load_pickrate_data()
    pickrate_dict = compute_player_pickrates(pickrate_df) if pickrate_df is not None else {}
    team_pop = compute_team_popularity(pickrate_df, all_data) if pickrate_df is not None else {}

    played = all_data[all_data["P?"] == 1]
    team_wr = compute_team_win_rates(played)
    schedule_opp_strength = load_schedule_opponent_strength(team_wr)
    player_avg_ppm = compute_player_avg_ppm(played)

    print(f"  Training data: {len(played)} played games, {played['Player'].nunique()} players")
    print(f"  Manual prices: {len(manual_df)} players, mean={manual_df['Price'].mean():.3f}")
    print(f"  Pickrate data: {len(pickrate_dict)} players")

    # --- Step 2: Generate prices ---
    print("\n[2/4] Running v3.1 pricing algorithm...")
    v3_df = generate_v3_prices(
        all_data, manual_df, pickrate_dict, team_pop,
        team_wr, schedule_opp_strength
    )
    print(f"  Generated prices for {len(v3_df)} players")
    print(f"  Mean: {v3_df['generated_vp'].mean():.3f}, Median: {v3_df['generated_vp'].median():.2f}")

    save_cols = ["Player", "Team", "Position", "generated_vp", "games_total",
                 "games_2026", "uncertainty"]
    v3_df[save_cols].sort_values("generated_vp", ascending=False).to_csv(
        os.path.join(DIR, "generated_prices_2026_v3.csv"), index=False
    )

    combined = manual_df.copy()
    gen_map = dict(zip(v3_df["Player"], v3_df["generated_vp"]))
    combined["generated_vp"] = combined["Player"].map(gen_map).fillna(combined["Price"])

    role_map = {}
    for _, row in manual_df.iterrows():
        role_map[row["Player"].lower()] = row.get("Position", "I")

    # --- Step 3: Team recommendations ---
    print("\n[3/4] Running team recommendations...")

    manual_gw = run_gw_recommendations(
        combined, "Price", player_avg_ppm, team_wr, role_map, label="Manual"
    )
    generated_gw = run_gw_recommendations(
        combined, "generated_vp", player_avg_ppm, team_wr, role_map, label="Generated"
    )

    # --- Step 4: Create outputs ---
    print("\n[4/4] Creating outputs...")
    excel_path = create_excel(manual_df, v3_df, manual_gw, generated_gw, role_map)
    doc_path = create_documentation(manual_df, v3_df)

    # --- Verification ---
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    merged = v3_df.merge(manual_df[["Player", "Price"]], on="Player")
    merged["abs_diff"] = (merged["generated_vp"] - merged["Price"]).abs()
    print(f"\n  Generated mean: {v3_df['generated_vp'].mean():.3f} (target: {TARGET_MEAN:.3f})")
    print(f"  MAE vs manual: {merged['abs_diff'].mean():.2f}")
    print(f"  Correlation: {merged['generated_vp'].corr(merged['Price']):.3f}")
    print(f"  Within 1 VP: {(merged['abs_diff']<=1).sum()}/{len(merged)}")
    print(f"  Within 2 VP: {(merged['abs_diff']<=2).sum()}/{len(merged)}")

    for label, gw_results in [("Manual", manual_gw), ("Generated", generated_gw)]:
        print(f"\n  {label} recommendations:")
        for gw_idx, result in enumerate(gw_results):
            if result is None:
                print(f"    GW{gw_idx+1}: FAILED")
                continue
            amer_count = sum(1 for p in result["players"]
                           if get_team_region_normalized(p["Team"]) == "AMER")
            zero_pts = [p["Player"] for p in result["players"] if p["gw_pts"] == 0]
            status = "OK"
            if zero_pts and gw_idx == 5:
                status = f"WARNING: {len(zero_pts)} non-AMER scoring 0"
            print(f"    GW{gw_idx+1}: {result['expected_pts_with_igl']:.1f} pts, "
                  f"AMER={amer_count}/11, VP={result['total_vp']:.1f}, {status}")

    print(f"\n  Excel: {os.path.exists(excel_path)}")
    print(f"  Doc: {os.path.exists(doc_path)}")
    print("\n  DONE.")


if __name__ == "__main__":
    main()
