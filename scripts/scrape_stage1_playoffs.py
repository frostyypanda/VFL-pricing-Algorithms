"""Scrape Stage 1 playoff matches for AMER/EMEA/PAC + all China Stage 1.

Writes data/stage1_playoffs_results.json (AMER/EMEA/PAC playoffs) and
data/stage1_china_results.json (all China group + playoff matches).

Match IDs sourced from vlr.gg event pages:
  AMER playoffs: event 2860
  EMEA playoffs: event 2863
  PAC  playoffs: event 2775
  China full:    event 2864
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


PLAYOFFS = {
    "AMER": [
        (660369, "UR1"), (660370, "UR1"),
        (660371, "USF"), (660372, "USF"),
        (660375, "LR1"), (660376, "LR1"),
        (660377, "LR2"), (660378, "LR2"),
        (660373, "UF"),
        (660379, "LR3"),
        (660380, "LF"),
        (660374, "GF"),
    ],
    "EMEA": [
        (660381, "UR1"), (660382, "UR1"),
        (660383, "USF"), (660384, "USF"),
        (660387, "LR1"), (660388, "LR1"),
        (660389, "LR2"), (660390, "LR2"),
        (660385, "UF"),
        (660391, "LR3"),
        (660392, "LF"),
        (660386, "GF"),
    ],
    "PAC": [
        (666488, "UR1"), (666489, "UR1"),
        (666490, "USF"), (666491, "USF"),
        (666494, "LR1"), (666495, "LR1"),
        (666496, "LR2"), (666497, "LR2"),
        (666492, "UF"),
        (666498, "LR3"),
        (666499, "LF"),
        (666493, "GF"),
    ],
}


CHINA_GROUP = [
    (642914, "G1"), (642915, "G1"), (642916, "G1"), (642917, "G1"),
    (642918, "G1"), (642919, "G1"),
    (642920, "G2"), (642921, "G2"), (642922, "G2"), (642923, "G2"),
    (642924, "G2"), (642925, "G2"),
    (642927, "G3"), (642928, "G3"), (642929, "G3"), (642930, "G3"),
    (642931, "G3"), (642932, "G3"),
    (642933, "G4"), (642934, "G4"), (642935, "G4"), (642936, "G4"),
    (642937, "G4"), (642938, "G4"),
    (642939, "G5"), (642940, "G5"), (642941, "G5"), (642942, "G5"),
    (642943, "G5"), (642944, "G5"),
    (642947, "SR1"), (642946, "SR1"), (642948, "SR1"), (642945, "SR1"),
]

CHINA_PLAYOFFS = [
    (659474, "UQF"), (659475, "UQF"), (659476, "UQF"), (659477, "UQF"),
    (659482, "LR1"), (659483, "LR1"),
    (659478, "USF"), (659479, "USF"),
    (659485, "LR2"), (659484, "LR2"),
    (659480, "UF"),
    (659486, "LR3"),
    (659487, "LF"),
    (659481, "GF"),
]


def scrape_match(mid, region, game_label):
    print(f"  Scraping {mid} ({region} {game_label})...")
    soup = fetch_match(mid)
    team1, team2 = parse_match_header(soup)
    map_scores = parse_map_scores(soup)
    per_map = parse_per_map_stats(soup)
    agg = parse_all_maps_stats(soup)
    time.sleep(1.0)
    perf_soup = fetch_performance(mid)
    multikills = parse_multikills(perf_soup)
    return {
        "match_id": mid, "region": region, "game": game_label,
        "team1": team1, "team2": team2,
        "map_scores": map_scores, "per_map": per_map,
        "aggregate": agg, "multikills": multikills,
    }


def scrape_set(jobs, delay=1.2):
    """Scrape a list of (mid, region, game_label) tuples."""
    matches = []
    for i, (mid, region, game) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}]", end=" ")
        try:
            matches.append(scrape_match(mid, region, game))
        except Exception as e:
            print(f"  ERROR on {mid}: {e}")
        if i < len(jobs) - 1:
            time.sleep(delay)
    return matches


def build_player_points(matches):
    player_points = {}
    for m in matches:
        pts = compute_match_points(m)
        for name, d in pts.items():
            key = f"{name}_{m['match_id']}"
            player_points[key] = {
                **d, "region": m["region"], "match_id": m["match_id"],
                "game": m["game"],
            }
    return player_points


def main():
    # Stage 1 playoffs for AMER/EMEA/PAC
    playoff_jobs = []
    for region, ids in PLAYOFFS.items():
        for mid, game in ids:
            playoff_jobs.append((mid, region, game))
    print(f"=== Scraping {len(playoff_jobs)} Stage 1 playoff matches ===")
    playoff_matches = scrape_set(playoff_jobs)
    playoff_pts = build_player_points(playoff_matches)
    out1 = os.path.join("data", "stage1_playoffs_results.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump({"matches": playoff_matches, "player_points": playoff_pts},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out1}: {len(playoff_matches)} matches, "
          f"{len(playoff_pts)} player entries")

    # China full Stage 1 (group + seeding + playoffs)
    china_jobs = []
    for mid, game in CHINA_GROUP:
        china_jobs.append((mid, "CN", game))
    for mid, game in CHINA_PLAYOFFS:
        china_jobs.append((mid, "CN", game))
    print(f"\n=== Scraping {len(china_jobs)} China Stage 1 matches ===")
    china_matches = scrape_set(china_jobs)
    china_pts = build_player_points(china_matches)
    out2 = os.path.join("data", "stage1_china_results.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump({"matches": china_matches, "player_points": china_pts},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out2}: {len(china_matches)} matches, "
          f"{len(china_pts)} player entries")


if __name__ == "__main__":
    main()
