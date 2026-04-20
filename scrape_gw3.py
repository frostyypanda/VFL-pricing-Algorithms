"""Scrape GW3 matches: AMER W2 + EMEA W3 + PAC W3. Writes data/w3_vlr_results.json."""
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

# GW3 in VFL = AM W2 + EMEA W3 + PAC W3
GW3_MATCHES = {
    "AMER": [645480, 645481, 645482, 645483, 645484, 645485],
    "EMEA": [644722, 644723, 644724, 644725, 644726, 644727],
    "PAC":  [644665, 644666, 644667, 644668, 644669, 644670],
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
    all_ids = [(r, mid) for r, ids in GW3_MATCHES.items() for mid in ids]
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

    out_path = os.path.join("data", "w3_vlr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"matches": matches, "player_points": player_points},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}: {len(matches)} matches, {len(player_points)} entries")


if __name__ == "__main__":
    main()
