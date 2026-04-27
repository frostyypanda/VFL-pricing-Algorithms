"""Scrape GW4 matches: AMER W3 + EMEA W4 + PAC W4. Writes data/w4_vlr_results.json."""
import sys
import os
import io
import json
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2.vlr_scraper import (
    fetch_match, fetch_performance, parse_match_header, parse_map_scores,
    parse_per_map_stats, parse_all_maps_stats, parse_multikills,
)
from v2.vfl_points import compute_match_points

# GW4 in VFL = AMER W3 + EMEA W4 + PAC W4 (verified via vlr.gg, 2026-04-27)
GW4_MATCHES = {
    "AMER": [645486, 645487, 645488, 645489, 645490, 645491],
    "EMEA": [644728, 644729, 644730, 644731, 644732, 644733],
    "PAC":  [644671, 644672, 644673, 644674, 644675, 644676],
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
    all_ids = [(r, mid) for r, ids in GW4_MATCHES.items() for mid in ids]
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

    out_path = os.path.join("data", "w4_vlr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"matches": matches, "player_points": player_points},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}: {len(matches)} matches, {len(player_points)} entries")


if __name__ == "__main__":
    main()
