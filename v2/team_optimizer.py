"""Full-season team optimizer via Integer Linear Programming.

Single ILP over all 6 gameweeks. No heuristics.
Maximizes total expected points (incl. IGL doubling) subject to
budget, role, team, and transfer constraints.
"""
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value,
    PULP_CBC_CMD,
)
from .constants import (
    VP_MIN, VP_MAX, BUDGET, SQUAD_SIZE,
    ROLE_SLOTS, WILDCARD_SLOTS, MAX_PER_TEAM, MAX_TRANSFERS, NUM_GWS,
)

SLOT_ORDER = [
    "D1", "D2", "I1", "I2", "C1", "C2", "S1", "S2", "W1", "W2", "W3",
]


def optimize_season(ep_matrix, prices):
    """Find globally optimal team + transfer plan across all 6 GWs.

    Returns list of 6 dicts, one per GW.
    """
    players = ep_matrix.to_dict("records")
    n = len(players)
    gws = range(1, NUM_GWS + 1)

    prob = LpProblem("VFL_Season", LpMaximize)
    solver = PULP_CBC_CMD(msg=0, timeLimit=120, threads=1)

    # --- Variables ---
    R = {(i, g): LpVariable(f"r_{i}_{g}", cat="Binary")
         for i in range(n) for g in gws}
    SD = {(i, g): LpVariable(f"sd_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SC = {(i, g): LpVariable(f"sc_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SI = {(i, g): LpVariable(f"si_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SS = {(i, g): LpVariable(f"ss_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SW = {(i, g): LpVariable(f"sw_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    IGL = {(i, g): LpVariable(f"igl_{i}_{g}", cat="Binary")
           for i in range(n) for g in gws}
    TIN = {(i, g): LpVariable(f"tin_{i}_{g}", cat="Binary")
           for i in range(n) for g in gws}

    # --- Objective: max total pts + IGL bonus ---
    obj = []
    for g in gws:
        col = f"GW{g}"
        for i in range(n):
            pts = players[i].get(col, 0)
            obj.append(R[i, g] * pts)
            obj.append(IGL[i, g] * pts)
    prob += lpSum(obj)

    # --- Constraints (per GW) ---
    teams_idx = {}
    for i in range(n):
        teams_idx.setdefault(players[i]["Team"], []).append(i)

    for g in gws:
        prob += lpSum(R[i, g] for i in range(n)) == SQUAD_SIZE
        prob += lpSum(
            R[i, g] * prices.get(players[i]["Player"], VP_MAX)
            for i in range(n)
        ) <= BUDGET

        for idxs in teams_idx.values():
            prob += lpSum(R[i, g] for i in idxs) <= MAX_PER_TEAM

        for i in range(n):
            prob += SD[i, g] + SC[i, g] + SI[i, g] + SS[i, g] + SW[i, g] == R[i, g]
            role = players[i].get("Role", "")
            if role != "D":
                prob += SD[i, g] == 0
            if role != "C":
                prob += SC[i, g] == 0
            if role != "I":
                prob += SI[i, g] == 0
            if role != "S":
                prob += SS[i, g] == 0

        prob += lpSum(SD[i, g] for i in range(n)) == ROLE_SLOTS["D"]
        prob += lpSum(SC[i, g] for i in range(n)) == ROLE_SLOTS["C"]
        prob += lpSum(SI[i, g] for i in range(n)) == ROLE_SLOTS["I"]
        prob += lpSum(SS[i, g] for i in range(n)) == ROLE_SLOTS["S"]
        prob += lpSum(SW[i, g] for i in range(n)) == WILDCARD_SLOTS

        prob += lpSum(IGL[i, g] for i in range(n)) == 1
        for i in range(n):
            prob += IGL[i, g] <= R[i, g]

    # --- Transfer constraints ---
    for g in gws:
        for i in range(n):
            if g == 1:
                prob += TIN[i, 1] == R[i, 1]
            else:
                prob += TIN[i, g] >= R[i, g] - R[i, g - 1]
                prob += TIN[i, g] <= R[i, g]
                prob += TIN[i, g] <= 1 - R[i, g - 1] + (1 - R[i, g])
        if g >= 2:
            prob += lpSum(TIN[i, g] for i in range(n)) <= MAX_TRANSFERS

    # --- Position-lock: if a player stays, their slot type stays ---
    for i in range(n):
        for g in range(2, NUM_GWS + 1):
            for sv in [SD, SC, SI, SS, SW]:
                prob += sv[i, g] >= sv[i, g - 1] + R[i, g] - 1

    # --- Solve ---
    prob.solve(solver)
    if LpStatus[prob.status] != "Optimal":
        return None

    return _extract(players, prices, R, SD, SC, SI, SS, SW, IGL, TIN, n, gws)


def optimize_with_locked_gw1(ep_matrix, prices, locked_players, locked_slots):
    """Optimize GW2-6 given a locked GW1 team."""
    players = ep_matrix.to_dict("records")
    n = len(players)
    gws = range(1, NUM_GWS + 1)
    player_idx = {p["Player"]: i for i, p in enumerate(players)}

    prob = LpProblem("VFL_Transfer", LpMaximize)
    solver = PULP_CBC_CMD(msg=0, timeLimit=120, threads=1)

    R = {(i, g): LpVariable(f"r_{i}_{g}", cat="Binary")
         for i in range(n) for g in gws}
    SD = {(i, g): LpVariable(f"sd_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SC = {(i, g): LpVariable(f"sc_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SI = {(i, g): LpVariable(f"si_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SS = {(i, g): LpVariable(f"ss_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    SW = {(i, g): LpVariable(f"sw_{i}_{g}", cat="Binary")
          for i in range(n) for g in gws}
    IGL = {(i, g): LpVariable(f"igl_{i}_{g}", cat="Binary")
           for i in range(n) for g in gws}
    TIN = {(i, g): LpVariable(f"tin_{i}_{g}", cat="Binary")
           for i in range(n) for g in gws}

    # Lock GW1
    locked_set = set(locked_players)
    for i in range(n):
        name = players[i]["Player"]
        if name in locked_set:
            prob += R[i, 1] == 1
            st = locked_slots[name][0]
            prob += SD[i, 1] == (1 if st == "D" else 0)
            prob += SC[i, 1] == (1 if st == "C" else 0)
            prob += SI[i, 1] == (1 if st == "I" else 0)
            prob += SS[i, 1] == (1 if st == "S" else 0)
            prob += SW[i, 1] == (1 if st == "W" else 0)
        else:
            prob += R[i, 1] == 0

    # Same objective and constraints as optimize_season
    obj = []
    for g in gws:
        col = f"GW{g}"
        for i in range(n):
            pts = players[i].get(col, 0)
            obj.append(R[i, g] * pts)
            obj.append(IGL[i, g] * pts)
    prob += lpSum(obj)

    teams_idx = {}
    for i in range(n):
        teams_idx.setdefault(players[i]["Team"], []).append(i)

    for g in gws:
        prob += lpSum(R[i, g] for i in range(n)) == SQUAD_SIZE
        prob += lpSum(
            R[i, g] * prices.get(players[i]["Player"], VP_MAX)
            for i in range(n)
        ) <= BUDGET
        for idxs in teams_idx.values():
            prob += lpSum(R[i, g] for i in idxs) <= MAX_PER_TEAM
        for i in range(n):
            prob += SD[i, g] + SC[i, g] + SI[i, g] + SS[i, g] + SW[i, g] == R[i, g]
            role = players[i].get("Role", "")
            if role != "D":
                prob += SD[i, g] == 0
            if role != "C":
                prob += SC[i, g] == 0
            if role != "I":
                prob += SI[i, g] == 0
            if role != "S":
                prob += SS[i, g] == 0
        prob += lpSum(SD[i, g] for i in range(n)) == ROLE_SLOTS["D"]
        prob += lpSum(SC[i, g] for i in range(n)) == ROLE_SLOTS["C"]
        prob += lpSum(SI[i, g] for i in range(n)) == ROLE_SLOTS["I"]
        prob += lpSum(SS[i, g] for i in range(n)) == ROLE_SLOTS["S"]
        prob += lpSum(SW[i, g] for i in range(n)) == WILDCARD_SLOTS
        prob += lpSum(IGL[i, g] for i in range(n)) == 1
        for i in range(n):
            prob += IGL[i, g] <= R[i, g]

    for g in gws:
        for i in range(n):
            if g == 1:
                prob += TIN[i, 1] == R[i, 1]
            else:
                prob += TIN[i, g] >= R[i, g] - R[i, g - 1]
                prob += TIN[i, g] <= R[i, g]
                prob += TIN[i, g] <= 1 - R[i, g - 1] + (1 - R[i, g])
        if g >= 2:
            prob += lpSum(TIN[i, g] for i in range(n)) <= MAX_TRANSFERS

    for i in range(n):
        for g in range(2, NUM_GWS + 1):
            for sv in [SD, SC, SI, SS, SW]:
                prob += sv[i, g] >= sv[i, g - 1] + R[i, g] - 1

    prob.solve(solver)
    if LpStatus[prob.status] != "Optimal":
        return None
    return _extract(players, prices, R, SD, SC, SI, SS, SW, IGL, TIN, n, gws)


def _extract(players, prices, R, SD, SC, SI, SS, SW, IGL, TIN, n, gws):
    """Extract solution into per-GW result dicts."""
    results = []
    for g in gws:
        col = f"GW{g}"
        selected, slots, igl_name, transfers = [], {}, None, []

        for i in range(n):
            if value(R[i, g]) > 0.5:
                p = dict(players[i])
                p["price"] = prices.get(p["Player"], VP_MAX)
                p["gw_pts"] = p.get(col, 0)
                selected.append(p)

                st = ("D" if value(SD[i, g]) > 0.5 else
                      "C" if value(SC[i, g]) > 0.5 else
                      "I" if value(SI[i, g]) > 0.5 else
                      "S" if value(SS[i, g]) > 0.5 else "W")
                slots[p["Player"]] = st

                if value(IGL[i, g]) > 0.5:
                    igl_name = p["Player"]
                if g >= 2 and value(TIN[i, g]) > 0.5:
                    transfers.append(("?", p["Player"], st))

        numbered = _number_slots(selected, slots)

        # Match transfer ins with outs
        if g >= 2 and transfers:
            prev = results[g - 2]
            prev_names = {p["Player"] for p in prev["players"]}
            curr_names = {p["Player"] for p in selected}
            outs = prev_names - curr_names
            out_by_slot = {}
            for name in outs:
                st = prev["slots"][name][0]
                out_by_slot.setdefault(st, []).append(name)
            named = []
            for _, in_name, st in transfers:
                out_list = out_by_slot.get(st, [])
                out_name = out_list.pop(0) if out_list else "?"
                named.append((out_name, in_name, numbered.get(in_name, st)))
            transfers = named

        vp = sum(p["price"] for p in selected)
        pts = sum(p["gw_pts"] for p in selected)
        igl_pts = next(
            (p["gw_pts"] for p in selected if p["Player"] == igl_name), 0
        )

        results.append({
            "players": selected,
            "total_vp": round(vp, 1),
            "expected_pts": round(pts, 2),
            "expected_pts_with_igl": round(pts + igl_pts, 2),
            "igl": igl_name,
            "slots": numbered,
            "transfers": transfers,
        })
    return results


def _number_slots(players, slot_types):
    counters = {"D": 0, "C": 0, "I": 0, "S": 0, "W": 0}
    numbered = {}
    for p in sorted(players, key=lambda p: p.get("gw_pts", 0), reverse=True):
        st = slot_types[p["Player"]]
        counters[st] += 1
        numbered[p["Player"]] = f"{st}{counters[st]}"
    return numbered
