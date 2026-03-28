"""
VLR.gg 2024 VCT Data Scraper -> VFL CSV Format

Scrapes all 2024 VCT match data from vlr.gg for Americas, EMEA, Pacific
(excluding China) and outputs a CSV matching the 2025 VFL.csv format.

Usage:
    python scrape_vlr_2024.py              # Full scrape (parallel)
    python scrape_vlr_2024.py --test       # Test with 2 matches per event
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ── Config ──

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
BASE_URL = "https://www.vlr.gg"
MAX_WORKERS = 4          # concurrent event scrapers
REQUEST_DELAY = 0.75     # seconds between requests per worker
MAX_RETRIES = 3

# Thread-local rate limiting
_thread_local = threading.local()
_print_lock = threading.Lock()

# 2024 VCT events (Americas, EMEA, Pacific -- no China)
EVENTS = {
    "Kickoff": [
        {"id": 1923, "slug": "champions-tour-2024-americas-kickoff", "region": "Americas"},
        {"id": 1925, "slug": "champions-tour-2024-emea-kickoff", "region": "EMEA"},
        {"id": 1924, "slug": "champions-tour-2024-pacific-kickoff", "region": "Pacific"},
    ],
    "Madrid": [
        {"id": 1921, "slug": "champions-tour-2024-masters-madrid", "region": "International"},
    ],
    "Stage 1": [
        {"id": 2004, "slug": "champions-tour-2024-americas-stage-1", "region": "Americas"},
        {"id": 1998, "slug": "champions-tour-2024-emea-stage-1", "region": "EMEA"},
        {"id": 2002, "slug": "champions-tour-2024-pacific-stage-1", "region": "Pacific"},
    ],
    "Shanghai": [
        {"id": 1999, "slug": "champions-tour-2024-masters-shanghai", "region": "International"},
    ],
    "Stage 2": [
        {"id": 2095, "slug": "champions-tour-2024-americas-stage-2", "region": "Americas"},
        {"id": 2094, "slug": "champions-tour-2024-emea-stage-2", "region": "EMEA"},
        {"id": 2005, "slug": "champions-tour-2024-pacific-stage-2", "region": "Pacific"},
    ],
    "Champions": [
        {"id": 2097, "slug": "champions-tour-2024-champions", "region": "International"},
    ],
}


def log(msg):
    with _print_lock:
        print(msg, flush=True)


# ── HTTP with per-thread rate limiting ──

def fetch_soup(url):
    """Fetch a URL with per-thread rate limiting and retries."""
    last = getattr(_thread_local, "last_request_time", 0)
    elapsed = time.time() - last
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            _thread_local.last_request_time = time.time()
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ── Parsing helpers ──

def parse_stat_value(cell):
    """Extract the 'both sides' value from a mod-stat cell."""
    both_span = cell.find("span", class_="mod-both")
    if both_span:
        text = both_span.get_text(strip=True).replace("%", "")
        try:
            return float(text)
        except ValueError:
            return 0.0
    sq = cell.find("span", class_="stats-sq")
    if sq:
        text = sq.get_text(strip=True).replace("%", "")
        try:
            return float(text)
        except ValueError:
            return 0.0
    return 0.0


def parse_player_row(tr):
    """Parse a player row from a wf-table-inset.mod-overview table."""
    cells = tr.find_all("td")
    if len(cells) < 10:
        return None

    player_cell = cells[0]
    if "mod-player" not in (player_cell.get("class") or []):
        return None

    name_div = player_cell.find("div", class_="text-of")
    player_name = name_div.get_text(strip=True) if name_div else player_cell.get_text(strip=True)

    team_tag = player_cell.find("span", class_="ge-text-light")
    team_name = team_tag.get_text(strip=True) if team_tag else ""
    player_name = player_name.replace(team_name, "").strip()

    return {
        "player": player_name,
        "team": team_name,
        "rating": parse_stat_value(cells[2]),
        "kills": int(parse_stat_value(cells[4])),
        "deaths": int(parse_stat_value(cells[5])),
        "assists": int(parse_stat_value(cells[6])),
    }


def get_event_match_ids(event_id, event_slug):
    """Fetch all match IDs for an event."""
    url = f"{BASE_URL}/event/matches/{event_id}/{event_slug}/?series_id=all"
    soup = fetch_soup(url)

    match_ids = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parts = href.strip("/").split("/")
        if len(parts) >= 2 and parts[0].isdigit() and "vs" in href:
            mid = int(parts[0])
            if mid not in match_ids:
                match_ids.append(mid)
    return match_ids


def parse_match(match_id):
    """Scrape a single match: overview + performance tab."""

    url = f"{BASE_URL}/{match_id}"
    soup = fetch_soup(url)

    result = {
        "match_id": match_id,
        "teams": [],
        "match_scores": [],
        "maps": [],
        "player_overall_ratings": {},
        "multikills": {},
    }

    # Team names
    for td in soup.find_all("div", class_="match-header-link-name"):
        name = td.find("div", class_="wf-title-med")
        if name:
            result["teams"].append(name.get_text(strip=True))

    # Overall match score
    header_vs = soup.find("div", class_="match-header-vs-score")
    if header_vs:
        for s in header_vs.find_all("span"):
            t = s.get_text(strip=True)
            if t.isdigit():
                result["match_scores"].append(int(t))

    # Match date
    date_div = soup.find("div", class_="moment-tz-convert")
    if date_div:
        result["date"] = date_div.get("data-utc-ts", "")

    # Map nav items
    map_navs = soup.find_all("div", class_="vm-stats-gamesnav-item")
    maps_info = []
    for nav in map_navs:
        gid = nav.get("data-game-id")
        if gid and gid != "all" and nav.get("data-disabled") != "1":
            map_name = re.sub(r"^\d+", "", nav.get_text(strip=True)).strip()
            maps_info.append({"game_id": gid, "map_name": map_name})

    # Per-map stats
    for mi in maps_info:
        gid = mi["game_id"]
        game_div = soup.find("div", class_="vm-stats-game", attrs={"data-game-id": gid})
        if not game_div:
            continue

        header = game_div.find("div", class_="vm-stats-game-header")
        scores = []
        if header:
            for s in header.find_all("div", class_="score"):
                try:
                    scores.append(int(s.get_text(strip=True)))
                except ValueError:
                    pass

        tables = game_div.find_all("table", class_="mod-overview")
        players = []
        team_idx = 0
        for table in tables:
            for tr in table.find_all("tr"):
                p = parse_player_row(tr)
                if p:
                    p["team_side"] = team_idx
                    players.append(p)
            team_idx += 1

        result["maps"].append({
            "game_id": gid,
            "map_name": mi["map_name"],
            "scores": scores,
            "players": players,
        })

    # Overall ratings from "all maps" view
    all_div = soup.find("div", class_="vm-stats-game", attrs={"data-game-id": "all"})
    if all_div:
        for table in all_div.find_all("table", class_="mod-overview"):
            for tr in table.find_all("tr"):
                p = parse_player_row(tr)
                if p:
                    result["player_overall_ratings"][p["player"]] = p["rating"]

    # ── Performance tab (multi-kills) ──
    perf_url = f"{BASE_URL}/{match_id}/?game=all&tab=performance"
    soup_perf = fetch_soup(perf_url)

    adv_tables = soup_perf.find_all("table", class_="mod-adv-stats")
    for table in adv_tables:
        parent = table.find_parent("div", class_="vm-stats-game")
        gid = parent.get("data-game-id") if parent else None
        if gid != "all":
            continue

        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells or len(cells) < 6:
                continue

            raw_name = cells[0].get_text(strip=True)
            player_name = raw_name
            for team in result["teams"]:
                if raw_name.endswith(team):
                    player_name = raw_name[:-len(team)].strip()
                    break

            mk = {}
            for ci in range(2, min(6, len(cells))):
                h_idx = ci
                if h_idx < len(ths):
                    text = cells[ci].get_text(strip=True)
                    m = re.match(r"(\d+)", text)
                    mk[ths[h_idx]] = int(m.group(1)) if m else 0

            result["multikills"][player_name] = mk

    return result


# ── VFL Scoring ──

def kill_threshold_points(kills_on_map):
    """VFL kill threshold points for a single map.
      0    kills -> -3
      1-4  kills -> -1
      5-9  kills ->  0
      10-14 kills -> +1
      15-19 kills -> +2
      20-24 kills -> +3
      25-29 kills -> +4
      30+   kills -> +1 per additional 5
    """
    if kills_on_map == 0:
        return -3
    elif kills_on_map < 5:
        return -1
    elif kills_on_map < 10:
        return 0
    elif kills_on_map < 15:
        return 1
    elif kills_on_map < 20:
        return 2
    elif kills_on_map < 25:
        return 3
    elif kills_on_map < 30:
        return 4
    else:
        return 4 + ((kills_on_map - 25) // 5)


def kill_bracket(kills):
    """Return the kill bracket (0, 5, 10, ... 50) for a kill count."""
    return min((kills // 5) * 5, 50)


def map_result_points(team_score, opp_score):
    """Team-based points for a single map.
    From CLAUDE.md + data analysis:
      Win           -> +1
      Win by 5+     -> +1 bonus
      Win by 10+    -> +2 bonus (replaces the 5+ bonus)
      Loss by 10+   -> -1
      Loss 13-0     -> -2 (extra harsh)
    """
    if team_score > opp_score:
        pts = 1
        diff = team_score - opp_score
        if diff >= 10:
            pts += 2
        elif diff >= 5:
            pts += 1
        return pts
    else:
        diff = opp_score - team_score
        if team_score == 0:
            return -2  # 13-0 loss
        elif diff >= 10:
            return -1
        return 0


def compute_match_vfl(match_data):
    """Compute VFL rows for all players in a match."""

    teams = match_data["teams"]
    maps = match_data["maps"]
    overall_ratings = match_data["player_overall_ratings"]
    multikills = match_data["multikills"]
    num_maps = len(maps)

    if not teams or not maps:
        return []

    # Match winner
    ms = match_data["match_scores"]
    match_winner = ""
    if len(ms) >= 2:
        if ms[0] > ms[1]:
            match_winner = teams[0]
        elif ms[1] > ms[0]:
            match_winner = teams[1] if len(teams) > 1 else ""

    # TOP1/2/3 by overall rating
    sorted_ratings = sorted(overall_ratings.items(), key=lambda x: x[1], reverse=True)
    top_bonuses = {}
    for i, (name, _) in enumerate(sorted_ratings[:3]):
        top_bonuses[name] = [3, 2, 1][i]

    # Rating threshold bonuses
    rating_thresh = {}
    for name, r in overall_ratings.items():
        if r >= 2.0:
            rating_thresh[name] = (1, 1, 1)
        elif r >= 1.75:
            rating_thresh[name] = (1, 1, 0)
        elif r >= 1.5:
            rating_thresh[name] = (1, 0, 0)
        else:
            rating_thresh[name] = (0, 0, 0)

    # Collect player -> team mapping
    player_teams = {}
    player_sides = {}
    for m in maps:
        for p in m["players"]:
            player_teams[p["player"]] = p["team"]
            player_sides[p["player"]] = p["team_side"]

    rows = []
    for player_name in player_teams:
        team_name = player_teams[player_name]
        side = player_sides[player_name]

        p_pts = 0
        t_pts = 0
        brackets = defaultdict(int)

        for m in maps:
            player_map = None
            for p in m["players"]:
                if p["player"] == player_name:
                    player_map = p
                    break
            if not player_map:
                continue

            p_pts += kill_threshold_points(player_map["kills"])
            b = kill_bracket(player_map["kills"])
            brackets[b] += 1

            scores = m["scores"]
            if len(scores) == 2:
                if side == 0:
                    t_pts += map_result_points(scores[0], scores[1])
                else:
                    t_pts += map_result_points(scores[1], scores[0])

        # Multi-kill bonuses (4K=+1, 5K=+3)
        mk = multikills.get(player_name, {})
        four_k = mk.get("4K", 0)
        five_k = mk.get("5K", 0)
        p_pts += four_k * 1 + five_k * 3

        # Rating bonuses
        tb = top_bonuses.get(player_name, 0)
        top1 = 1 if tb == 3 else 0
        top2 = 1 if tb >= 2 else 0
        top3 = 1 if tb >= 1 else 0
        p_pts += tb

        rt = rating_thresh.get(player_name, (0, 0, 0))
        p_pts += sum(rt)

        total_pts = t_pts + p_pts
        ppm = round(total_pts / num_maps, 2) if num_maps else 0

        full_team = teams[0] if side == 0 else (teams[1] if len(teams) > 1 else "")
        wl = "W" if full_team == match_winner else "L"

        row = {
            "Team": full_team,
            "Player": player_name,
            "Pts": total_pts,
            "T.Pts": t_pts,
            "P.Pts": p_pts,
            "PPM": ppm,
            "Adj.VP": "",
            "P?": 1,
            "0k": brackets.get(0, 0),
            "5k": brackets.get(5, 0),
            "10k": brackets.get(10, 0),
            "15k": brackets.get(15, 0),
            "20k": brackets.get(20, 0),
            "25k": brackets.get(25, 0),
            "30k": brackets.get(30, 0),
            "35k": brackets.get(35, 0),
            "40k": brackets.get(40, 0),
            "45k": brackets.get(45, 0),
            "50k": brackets.get(50, 0),
            "4ks": four_k,
            "5ks": five_k,
            "TOP3": top3,
            "TOP2": top2,
            "TOP1": top1,
            "1.5R2": rt[0],
            "1.75R2": rt[1],
            "2.0R2": rt[2],
            "PR Avg.": "",
            "W/L": wl,
            "Game Start VP": "",
            "Game End VP": "",
        }
        rows.append(row)

    return rows


# ── Scraping pipeline ──

def scrape_event(stage_name, event_info, test_mode=False):
    """Scrape all matches for one event. Runs in its own thread."""
    eid = event_info["id"]
    eslug = event_info["slug"]
    region = event_info["region"]
    label = f"[{stage_name}/{region}]"

    log(f"  {label} Fetching match list...")
    match_ids = get_event_match_ids(eid, eslug)
    log(f"  {label} Found {len(match_ids)} matches")

    if test_mode:
        match_ids = match_ids[:2]
        log(f"  {label} TEST MODE: limiting to 2 matches")

    all_rows = []
    for i, mid in enumerate(match_ids):
        log(f"  {label} Scraping match {i+1}/{len(match_ids)}: {mid}")
        try:
            match_data = parse_match(mid)
            rows = compute_match_vfl(match_data)

            for row in rows:
                row["Stage"] = stage_name
                row["Wk"] = ""
                row["Game"] = f"G{i+1}"
                row["match_id"] = mid
                row["date"] = match_data.get("date", "")
                row["region"] = region

            all_rows.extend(rows)
        except Exception as e:
            log(f"  {label} ERROR match {mid}: {e}")

    log(f"  {label} Done: {len(all_rows)} rows")
    return all_rows


def fill_missing_slots(all_rows):
    """Fill P?=0 rows for games a player didn't play in their stage."""
    player_stages = defaultdict(list)
    for row in all_rows:
        key = (row["Team"], row["Player"])
        player_stages[key].append(row)

    stage_games = defaultdict(set)
    for row in all_rows:
        stage_games[row["Stage"]].add(row["Game"])

    filled = list(all_rows)
    for (team, player), rows in player_stages.items():
        player_stages_set = set(r["Stage"] for r in rows)
        for stage in player_stages_set:
            player_games = set(r["Game"] for r in rows if r["Stage"] == stage)
            for game in stage_games[stage] - player_games:
                filled.append({
                    "Team": team, "Player": player,
                    "Stage": stage, "Wk": "", "Game": game,
                    "Pts": 0, "T.Pts": 0, "P.Pts": 0, "PPM": 0,
                    "Adj.VP": "", "P?": 0,
                    "0k": 0, "5k": 0, "10k": 0, "15k": 0, "20k": 0,
                    "25k": 0, "30k": 0, "35k": 0, "40k": 0, "45k": 0, "50k": 0,
                    "4ks": 0, "5ks": 0,
                    "TOP3": 0, "TOP2": 0, "TOP1": 0,
                    "1.5R2": 0, "1.75R2": 0, "2.0R2": 0,
                    "PR Avg.": "", "W/L": "L",
                    "Game Start VP": "", "Game End VP": "",
                })
    return filled


def build_csv(all_rows, output_path):
    """Build CSV matching 2025 VFL.csv column order."""
    rows_out = []
    for row in all_rows:
        rows_out.append({
            "": row["Team"],
            " ": row["Player"],
            "Stage": row["Stage"],
            "Wk": row.get("Wk", ""),
            "Game": row["Game"],
            "Pts": row["Pts"],
            "T.Pts": row["T.Pts"],
            "P.Pts": row["P.Pts"],
            "PPM": row["PPM"],
            "Adj.VP": row.get("Adj.VP", ""),
            "P?": row["P?"],
            "0k": row["0k"], "5k": row["5k"], "10k": row["10k"],
            "15k": row["15k"], "20k": row["20k"], "25k": row["25k"],
            "30k": row["30k"], "35k": row["35k"], "40k": row["40k"],
            "45k": row["45k"], "50k": row["50k"],
            "4ks": row["4ks"], "5ks": row["5ks"],
            "TOP3": row["TOP3"], "TOP2": row["TOP2"], "TOP1": row["TOP1"],
            "1.5R2": row["1.5R2"], "1.75R2": row["1.75R2"], "2.0R2": row["2.0R2"],
            "PR Avg.": row.get("PR Avg.", ""),
            "W/L": row["W/L"],
            "Game Start VP": row.get("Game Start VP", ""),
            "Game End VP": row.get("Game End VP", ""),
        })

    df = pd.DataFrame(rows_out)

    stage_order = {"Kickoff": 0, "Madrid": 1, "Stage 1": 2, "Shanghai": 3, "Stage 2": 4, "Champions": 5}
    df["_stage_order"] = df["Stage"].map(stage_order).fillna(99)
    df["_game_num"] = df["Game"].str.extract(r"(\d+)").astype(float).fillna(0)
    df = df.sort_values(["", " ", "_stage_order", "_game_num"])
    df = df.drop(columns=["_stage_order", "_game_num"])

    df.columns = ["", ""] + list(df.columns[2:])
    df.to_csv(output_path, index=False, encoding="latin-1")
    log(f"\nWrote {len(df)} rows to {output_path}")


def main():
    test_mode = "--test" in sys.argv

    if test_mode:
        log("=" * 60)
        log("TEST MODE: 2 matches per event")
        log("=" * 60)

    # Flatten all events into a list of (stage, event_info) tasks
    tasks = []
    for stage_name, events in EVENTS.items():
        for event_info in events:
            tasks.append((stage_name, event_info))

    log(f"Scraping {len(tasks)} events with {MAX_WORKERS} parallel workers...\n")

    all_rows = []

    # Parallel scraping across events
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(scrape_event, stage, evt, test_mode): (stage, evt["region"])
            for stage, evt in tasks
        }

        for future in as_completed(futures):
            stage, region = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                log(f"  [{stage}/{region}] collected {len(rows)} rows (total so far: {len(all_rows)})")
            except Exception as e:
                log(f"  [{stage}/{region}] FAILED: {e}")

    log(f"\n{'='*60}")
    log(f"Total rows before filling slots: {len(all_rows)}")
    all_rows = fill_missing_slots(all_rows)
    log(f"Total rows after filling slots: {len(all_rows)}")

    output_name = "2024 VFL.csv" if not test_mode else "2024 VFL TEST.csv"
    output_path = os.path.join(os.path.dirname(__file__), output_name)
    build_csv(all_rows, output_path)

    df = pd.DataFrame(all_rows)
    played = df[df["P?"] == 1]
    log(f"\nSummary:")
    log(f"  Unique players: {played['Player'].nunique()}")
    log(f"  Unique teams: {played['Team'].nunique()}")
    log(f"  Stages: {sorted(played['Stage'].unique().tolist())}")
    log(f"  Avg points (played games): {played['Pts'].mean():.1f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    log(f"\nCompleted in {elapsed/60:.1f} minutes")
