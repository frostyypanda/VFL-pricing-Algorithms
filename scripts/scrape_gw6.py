"""Scrape GW6 matches: AMER W5 only (EMEA/PAC stage finished at GW5).
Writes data/w6_vlr_results.json.

AMER Stage 1 W5 match IDs sourced from vlr.gg event 2860 (verified 2026-05-13).
"""
import sys
import os
import io
import json
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.vlr_scraper import (
    fetch_match, fetch_performance, parse_match_header, parse_map_scores,
    parse_per_map_stats, parse_all_maps_stats, parse_multikills,
)
from v2.vfl_points import compute_match_points

GW6_MATCHES = {
    "AMER": [645498, 645499, 645500, 645501, 645502, 645503],
}


def scrape_match_full(region, mid):
    print(f"  Scraping {mid} ({region})...")
    soup = fetch_match(mid)
    team1, team2 = parse_match_header(soup)
    map_scores = parse_map_scores(soup)
    per_map = parse_per_map_stats(soup)
    agg = parse_all_maps_stats(soup)
    time.sleep(1.0)
    perf_soup = fetch_performance(mid)
    multikills = parse_multikills(perf_soup)
    return {
        "match_id": mid, "region": region,
        "team1": team1, "team2": team2,
        "map_scores": map_scores, "per_map": per_map,
        "aggregate": agg, "multikills": multikills,
    }


def main():
    all_ids = [(r, mid) for r, ids in GW6_MATCHES.items() for mid in ids]
    matches = []
    for i, (region, mid) in enumerate(all_ids):
        print(f"[{i+1}/{len(all_ids)}]", end=" ")
        matches.append(scrape_match_full(region, mid))
        if i < len(all_ids) - 1:
            time.sleep(1.2)

    player_points = {}
    for m in matches:
        pts = compute_match_points(m)
        for name, d in pts.items():
            key = f"{name}_{m['match_id']}"
            player_points[key] = {
                **d, "region": m["region"], "match_id": m["match_id"],
            }

    out_path = os.path.join("data", "w6_vlr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"matches": matches, "player_points": player_points},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}: {len(matches)} matches, {len(player_points)} entries")


if __name__ == "__main__":
    main()
