"""Generate GW6 transfer plan (final gameweek, AMER-only).

Approach:
  1. Load full historical data + calibrated EP model.
  2. Load W1..W5 VLR results — blend into EP.
  3. Run slot-locked ILP for GW6 only, with min_amer=11.
  4. Starting roster = post-GW5 XI (after the GW5 transfer plan was applied).
  5. Emit a markdown report.
"""
import sys
import os
import io
import json

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value

from v2.data_loader import load_all_data, load_manual_prices
from v2.expected_points import calibrate, compute_expected_pts
from v2.constants import BUDGET, MAX_TRANSFERS

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Post-GW5 XI: what the user's roster looks like after applying the GW5
# transfer plan (Rosé→mwzera, Patmen→Reduxx, Udotan→Cortezia).
# 8 AMER baseline; must reach 11 for GW6.
USER_TEAM = [
    ("D1", "Alym"),     ("D2", "Neon"),
    ("I1", "mwzera"),   ("I2", "Killua"),
    ("C1", "Veqaj"),    ("C2", "Zekken"),
    ("S1", "Free1ng"),  ("S2", "Tex"),
    ("W1", "Reduxx"),   ("W2", "Cortezia"), ("W3", "Saadhak"),
]


def load_week_results(path, our_players):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {p.lower(): p for p in our_players}
    result = {}
    for key, d in data["player_points"].items():
        name = key.rsplit("_", 1)[0]
        matched = lookup.get(name.lower())
        if not matched:
            continue
        result.setdefault(matched, []).append({
            "pts": d["pts"], "maps": d["maps"], "rating": d["rating"],
        })
    return result


def blend_actuals(ep, weeks):
    """Blend N weeks of actuals into EP BasePts and per-GW predictions."""
    df = ep.copy()
    alpha_per_week = 0.25

    for i, row in df.iterrows():
        player = row["Player"]
        prior = row["BasePts"]
        updated = prior
        for w in weeks:
            for weekdata in w.get(player, []):
                n_maps = max(weekdata["maps"], 1)
                observed_per_game = weekdata["pts"] / n_maps * 2.3
                updated = alpha_per_week * observed_per_game + (1 - alpha_per_week) * updated
        df.at[i, "BasePts"] = updated
        if prior > 0.1:
            ratio = updated / prior
            for gw in range(5, 7):
                col = f"GW{gw}"
                if col in row and row[col] > 0:
                    df.at[i, col] = row[col] * ratio
    return df


def build_starting_roster(ep, prices, spec):
    roster = []
    for slot, name in spec:
        row = ep[ep["Player"] == name]
        if len(row) == 0:
            print(f"  WARNING: player '{name}' not in EP; skipping")
            continue
        r = row.iloc[0]
        roster.append({
            "slot": slot, "Player": name, "Team": r["Team"],
            "Region": r["Region"], "Role": r["Role"],
            "VP": prices.get(name, 9.0),
        })
    return roster


def _slot_role(slot):
    ch = slot[0]
    return ch if ch in "DICS" else None


def optimize_gw(ep, prices, roster, gw, min_amer):
    """ILP: slot-locked, <=3 transfers, <=100 VP, <=2/team, IGL doubles.
    For AMER-only week (GW6) we zero non-AMER EP so IGL lands on AMER.
    """
    gw_col = f"GW{gw}"
    all_p = ep.to_dict("records")
    n = len(all_p)
    slots = [r["slot"] for r in roster]
    ns = len(slots)
    roster_names = {r["Player"] for r in roster}

    def ep_val(j):
        # AMER-only week: zero out non-AMER EP entirely.
        if all_p[j].get("Region") != "AMER":
            return 0.0
        return all_p[j][gw_col]

    def cur_ep_val(player_name):
        # For incumbent (not transferred out), same AMER-only filter.
        idx = p_idx.get(player_name)
        if idx is None:
            return 0.0
        return ep_val(idx)

    p_idx = {r["Player"]: i for i, r in enumerate(all_p)}

    prob = LpProblem(f"GW{gw}", LpMaximize)
    out = [LpVariable(f"out_{s}", cat="Binary") for s in range(ns)]
    inp = [[LpVariable(f"in_{s}_{j}", cat="Binary") for j in range(n)] for s in range(ns)]
    igl = [LpVariable(f"igl_{s}", cat="Binary") for s in range(ns)]
    sep = [LpVariable(f"sep_{s}", lowBound=0) for s in range(ns)]
    iep = [LpVariable(f"iep_{s}", lowBound=0) for s in range(ns)]
    BIG_M = 40.0

    prob += lpSum(sep[s] + iep[s] for s in range(ns))

    for s in range(ns):
        cur_ep = cur_ep_val(roster[s]["Player"])

        prob += sep[s] == (1 - out[s]) * cur_ep + lpSum(inp[s][j] * ep_val(j) for j in range(n))
        prob += lpSum(inp[s][j] for j in range(n)) == out[s]

        required = _slot_role(slots[s])
        for j in range(n):
            if required and all_p[j]["Role"] != required:
                prob += inp[s][j] == 0
            if all_p[j]["Player"] in roster_names:
                prob += inp[s][j] == 0

        prob += iep[s] <= sep[s]
        prob += iep[s] <= BIG_M * igl[s]
        prob += iep[s] >= sep[s] - BIG_M * (1 - igl[s])

    prob += lpSum(igl[s] for s in range(ns)) == 1
    prob += lpSum(out[s] for s in range(ns)) <= MAX_TRANSFERS

    for j in range(n):
        prob += lpSum(inp[s][j] for s in range(ns)) <= 1

    terms = []
    for s in range(ns):
        terms.append((1 - out[s]) * roster[s]["VP"])
        for j in range(n):
            terms.append(inp[s][j] * prices.get(all_p[j]["Player"], 9.0))
    prob += lpSum(terms) <= BUDGET

    amer_count = []
    for s in range(ns):
        if roster[s]["Region"] == "AMER":
            amer_count.append(1 - out[s])
        for j in range(n):
            if all_p[j].get("Region") == "AMER":
                amer_count.append(inp[s][j])
    if amer_count:
        prob += lpSum(amer_count) >= min_amer

    teams = set(r["Team"] for r in roster) | set(p["Team"] for p in all_p)
    for t in teams:
        tc = []
        for s in range(ns):
            if roster[s]["Team"] == t:
                tc.append(1 - out[s])
            for j in range(n):
                if all_p[j]["Team"] == t:
                    tc.append(inp[s][j])
        if tc:
            prob += lpSum(tc) <= 2

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))
    return _extract(roster, slots, out, inp, igl, all_p, prices, gw_col, gw)


def _extract(roster, slots, out, inp, igl, all_p, prices, gw_col, gw):
    new_roster = []
    team = []
    transfers = []
    igl_name = None

    for s in range(len(slots)):
        is_igl = value(igl[s]) > 0.5
        if value(out[s]) > 0.5:
            for j in range(len(all_p)):
                if value(inp[s][j]) > 0.5:
                    p = all_p[j]
                    transfers.append({"out": roster[s]["Player"], "in": p["Player"], "slot": slots[s]})
                    entry = {
                        "slot": slots[s], "Player": p["Player"], "Team": p["Team"],
                        "Region": p["Region"], "Role": p["Role"],
                        "VP": prices.get(p["Player"], 9.0),
                    }
                    new_roster.append(entry)
                    ep_show = p[gw_col] if p.get("Region") == "AMER" else 0.0
                    team.append({**entry, gw_col: ep_show, "IGL": is_igl})
                    if is_igl:
                        igl_name = p["Player"]
                    break
        else:
            r = roster[s]
            new_roster.append(dict(r))
            p_data = next((p for p in all_p if p["Player"] == r["Player"]), None)
            ep_gw = (p_data[gw_col] if (p_data and p_data.get("Region") == "AMER") else 0.0)
            team.append({**r, gw_col: ep_gw, "IGL": is_igl})
            if is_igl:
                igl_name = r["Player"]

    total_vp = sum(t["VP"] for t in team)
    total_pts = sum(t[gw_col] * (2 if t["IGL"] else 1) for t in team)
    return {
        "players": team, "total_vp": total_vp, "total_pts": total_pts,
        "igl": igl_name, "transfers": transfers, "roster": new_roster, "gw": gw,
    }


def render_team_plan(label, start_spec, plan, ep, prices):
    lines = [f"# {label} — GW6 Transfer Plan (Final GW, AMER-only)\n"]
    lines.append("## Starting Roster (post-GW5, blended through W1+W2+W3+W4+W5 actuals)\n")
    lines.append("| Slot | Player | Team | Region | Role | VP |")
    lines.append("|---|---|---|---|---|---|")
    total_start = 0.0
    amer_count = 0
    for slot, name in start_spec:
        row = ep[ep["Player"] == name].iloc[0]
        vp = prices.get(name, 9.0)
        total_start += vp
        if row["Region"] == "AMER":
            amer_count += 1
        lines.append(f"| {slot} | {name} | {row['Team']} | {row['Region']} | {row['Role']} | {vp:.1f} |")
    lines.append(f"\n**Total VP:** {total_start:.1f} | **AMER players:** {amer_count}/11\n")
    lines.append(f"**Constraint:** GW6 is AMER-only. Must hit 11 AMER, so the 3 non-AMER players "
                 f"(Killua/PAC, Veqaj/EMEA, Free1ng/PAC) all need replacing.\n")

    for gwp in plan:
        gw = gwp["gw"]
        gw_col = f"GW{gw}"
        lines.append(f"\n## GW{gw} — Plan\n")
        if gwp["transfers"]:
            lines.append("**Transfers:**\n")
            for t in gwp["transfers"]:
                out_p = ep[ep["Player"] == t["out"]].iloc[0]
                in_p = ep[ep["Player"] == t["in"]].iloc[0]
                out_vp = prices.get(t["out"], 9.0)
                in_vp = prices.get(t["in"], 9.0)
                lines.append(
                    f"- **{t['slot']}:** OUT {t['out']} ({out_p['Team']}/{out_p['Region']}, {out_vp:.1f}VP) "
                    f"→ IN {t['in']} ({in_p['Team']}/{in_p['Region']}, {in_vp:.1f}VP)"
                )
        else:
            lines.append("*No transfers suggested this week.*")

        lines.append(f"\n**Resulting XI (GW{gw}):**\n")
        lines.append("| Slot | Player | Team | Region | VP | Proj Pts | IGL |")
        lines.append("|---|---|---|---|---|---|---|")
        amer = 0
        for p in gwp["players"]:
            if p["Region"] == "AMER":
                amer += 1
            igl_marker = "★" if p["IGL"] else ""
            lines.append(
                f"| {p['slot']} | {p['Player']} | {p['Team']} | {p['Region']} "
                f"| {p['VP']:.1f} | {p[gw_col]:.1f} | {igl_marker} |"
            )
        lines.append(
            f"\n**VP:** {gwp['total_vp']:.1f}/100 | "
            f"**Projected pts (with IGL 2x):** {gwp['total_pts']:.1f} | "
            f"**IGL:** {gwp['igl']} | **AMER:** {amer}/11\n"
        )
    return "\n".join(lines)


def main():
    print("[1/5] Loading data & calibrating EP...")
    all_data = load_all_data()
    cal = calibrate(all_data)
    mp = load_manual_prices()
    train = all_data[all_data["P?"] == 1].copy()
    roster = {r["Player"]: {"team": r["Team"], "region": r["Region"], "role": r["Position"]}
              for _, r in mp.iterrows()}
    ep = compute_expected_pts(train, roster, cal)

    manual_map = dict(zip(mp["Player"], mp["Stage1_Price"]))
    ep["ManualVP"] = ep["Player"].map(manual_map)
    prices = dict(zip(ep["Player"], ep["ManualVP"]))

    print("[2/5] Loading W1+W2+W3+W4+W5 VLR actuals...")
    our_players = ep["Player"].tolist()
    weeks = []
    for wk in (1, 2, 3, 4, 5):
        path = os.path.join(DIR, "data", f"w{wk}_vlr_results.json")
        w = load_week_results(path, our_players)
        weeks.append(w)
        print(f"  W{wk} matched {len(w)} players")

    print("[3/5] Blending actuals into EP...")
    ep_updated = blend_actuals(ep, weeks)

    print("\n[4/5] Planning GW6 (AMER-only)...")
    starting = build_starting_roster(ep_updated, prices, USER_TEAM)
    if len(starting) != 11:
        print(f"  WARNING: roster has {len(starting)} players (expected 11)")

    start_amer = sum(1 for r in starting if r["Region"] == "AMER")
    print(f"  Starting AMER count: {start_amer}/11 → target 11")

    plan = [optimize_gw(ep_updated, prices, starting, 6, min_amer=11)]
    n_amer = sum(1 for x in plan[0]["roster"] if x["Region"] == "AMER")
    print(f"  GW6: {len(plan[0]['transfers'])} transfers, "
          f"VP={plan[0]['total_vp']:.1f}, EPts={plan[0]['total_pts']:.1f}, "
          f"AMER={n_amer}, IGL={plan[0]['igl']}")

    md = render_team_plan("MyTeam", USER_TEAM, plan, ep_updated, prices)
    out_path = os.path.join(DIR, "output", "GW6_transfer_plan.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Wrote {out_path}")

    print("\n[5/5] Done.")


if __name__ == "__main__":
    main()
