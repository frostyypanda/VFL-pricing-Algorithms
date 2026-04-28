"""Scrape GW2 matches: AM W1 + EMEA W2 + PAC W2. Writes data/w2_vlr_results.json."""
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

# GW2 in VFL = AM W1 + EMEA W2 + PAC W2
GW2_MATCHES = {
    "AMER": [645474, 645475, 645476, 645477, 645478, 645479],
    "EMEA": [644716, 644717, 644718, 644719, 644720, 644721],
    "PAC":  [644658, 644659, 644660, 644661, 644662, 644663],
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
    all_ids = [(r, mid) for r, ids in GW2_MATCHES.items() for mid in ids]
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

    out_path = os.path.join("data", "w2_vlr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"matches": matches, "player_points": player_points},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}: {len(matches)} matches, {len(player_points)} entries")


if __name__ == "__main__":
    main()
