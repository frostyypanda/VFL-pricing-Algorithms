"""Scrape GW5 matches: AMER W4 + EMEA W5 + PAC W5. Writes data/w5_vlr_results.json."""
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

# GW5 in VFL = AMER W4 + EMEA W5 + PAC W5 (verified via vlr.gg, 2026-05-05)
GW5_MATCHES = {
    "AMER": [645492, 645493, 645494, 645495, 645496, 645497],
    "EMEA": [644734, 644735, 644736, 644737, 644738, 644739],
    "PAC":  [644677, 644678, 644679, 644680, 644681, 644682],
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
    all_ids = [(r, mid) for r, ids in GW5_MATCHES.items() for mid in ids]
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

    out_path = os.path.join("data", "w5_vlr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"matches": matches, "player_points": player_points},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}: {len(matches)} matches, {len(player_points)} entries")


if __name__ == "__main__":
    main()
