"""Scrape VLR.gg match pages for W1 Stage 1 player stats."""
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (VFL Research Bot)"}

W1_MATCHES = {
    "EMEA": [644709, 644710, 644711, 644712, 644713, 644714],
    "PAC": [644652, 644653, 644654, 644655, 644656, 644657],
}


def fetch_match(match_id):
    """Fetch and parse a single VLR match page."""
    url = f"https://www.vlr.gg/{match_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_match_header(soup):
    """Extract team names and match score."""
    t1 = soup.select_one("a.match-header-link.mod-1 .wf-title-med")
    t2 = soup.select_one("a.match-header-link.mod-2 .wf-title-med")
    team1 = t1.get_text(strip=True) if t1 else "Team1"
    team2 = t2.get_text(strip=True) if t2 else "Team2"
    return team1, team2


def parse_map_scores(soup):
    """Extract per-map round scores and winners."""
    maps = []
    for game in soup.select("div.vm-stats-game"):
        gid = game.get("data-game-id", "")
        if gid == "all" or not gid:
            continue
        header = game.select_one(".vm-stats-game-header")
        if not header:
            continue
        teams = header.select(".team")
        if len(teams) < 2:
            continue
        s1_el = teams[0].select_one(".score")
        s2_el = teams[1].select_one(".score")
        s1 = int(s1_el.get_text(strip=True)) if s1_el else 0
        s2 = int(s2_el.get_text(strip=True)) if s2_el else 0
        t1_win = "mod-win" in (s1_el.get("class", []) if s1_el else [])
        maps.append({"score1": s1, "score2": s2, "team1_won": t1_win})
    return maps


def parse_per_map_stats(soup):
    """Extract per-map kills for each player (for kill bracket scoring)."""
    per_map = []
    for game in soup.select("div.vm-stats-game"):
        gid = game.get("data-game-id", "")
        if gid == "all" or not gid:
            continue
        tables = game.select("table.wf-table-inset.mod-overview")
        map_players = {"team1": [], "team2": []}
        for t_idx, table in enumerate(tables[:2]):
            side = "team1" if t_idx == 0 else "team2"
            for row in table.select("tbody tr"):
                player_cell = row.select_one("td.mod-player")
                if not player_cell:
                    continue
                name_el = player_cell.select_one(".text-of")
                name = name_el.get_text(strip=True) if name_el else "?"
                stats = row.select("td.mod-stat")
                kills = _get_stat(stats, 2)
                map_players[side].append({"name": name, "kills": kills})
        per_map.append(map_players)
    return per_map


def parse_all_maps_stats(soup):
    """Extract aggregate game stats (rating, kills, deaths)."""
    all_game = soup.select_one('div.vm-stats-game[data-game-id="all"]')
    if not all_game:
        return {"team1": [], "team2": []}
    tables = all_game.select("table.wf-table-inset.mod-overview")
    result = {"team1": [], "team2": []}
    for t_idx, table in enumerate(tables[:2]):
        side = "team1" if t_idx == 0 else "team2"
        for row in table.select("tbody tr"):
            player_cell = row.select_one("td.mod-player")
            if not player_cell:
                continue
            name_el = player_cell.select_one(".text-of")
            name = name_el.get_text(strip=True) if name_el else "?"
            stats = row.select("td.mod-stat")
            rating = _get_stat_float(stats, 0)
            kills = _get_stat(stats, 2)
            deaths = _get_stat(stats, 3)
            assists = _get_stat(stats, 4)
            result[side].append({
                "name": name, "rating": rating,
                "kills": kills, "deaths": deaths, "assists": assists,
            })
    return result


def _get_stat(stats, idx):
    """Get integer stat from mod-both span."""
    if idx >= len(stats):
        return 0
    both = stats[idx].select_one(".mod-both")
    if both:
        try:
            return int(both.get_text(strip=True))
        except ValueError:
            return 0
    return 0


def _get_stat_float(stats, idx):
    """Get float stat from mod-both span."""
    if idx >= len(stats):
        return 0.0
    both = stats[idx].select_one(".mod-both")
    if both:
        try:
            return float(both.get_text(strip=True))
        except ValueError:
            return 0.0
    return 0.0


def fetch_performance(match_id):
    """Fetch the Performance tab for a match."""
    url = f"https://www.vlr.gg/{match_id}/?game=all&tab=performance"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_multikills(soup):
    """Parse 4K and 5K counts from the Performance tab.

    Returns dict of {player_name: {"4k": int, "5k": int}}.
    """
    result = {}
    all_game = soup.select_one('div.vm-stats-game[data-game-id="all"]')
    if not all_game:
        return result
    tables = all_game.select("table.wf-table-inset.mod-adv-stats")
    for table in tables:
        for row in table.select("tr"):
            cells = row.select("td")
            if len(cells) < 6:
                continue
            team_div = cells[0].select_one(".team")
            if not team_div:
                continue
            tag = team_div.select_one(".team-tag")
            tag_text = tag.get_text(strip=True) if tag else ""
            name = team_div.get_text(strip=True).replace(tag_text, "").strip()
            fk = _parse_multikill_cell(cells[4])  # 4K column (index 4)
            ace = _parse_multikill_cell(cells[5])  # 5K column (index 5)
            result[name] = {"4k": fk, "5k": ace}
    return result


def _parse_multikill_cell(cell):
    """Parse a multi-kill cell value. Returns int."""
    sq = cell.select_one("div.stats-sq")
    if not sq or "mod-egg" in sq.get("class", []):
        return 0
    text = sq.get_text(strip=True)
    try:
        return int(text)
    except ValueError:
        return 0


def scrape_all_w1():
    """Scrape all W1 matches (overview + performance). Returns list."""
    matches = []
    all_ids = []
    for region, ids in W1_MATCHES.items():
        for mid in ids:
            all_ids.append((region, mid))

    for i, (region, mid) in enumerate(all_ids):
        print(f"  Scraping match {mid} ({region}) [{i+1}/{len(all_ids)}]...")
        soup = fetch_match(mid)
        team1, team2 = parse_match_header(soup)
        map_scores = parse_map_scores(soup)
        per_map = parse_per_map_stats(soup)
        agg = parse_all_maps_stats(soup)

        time.sleep(1.0)
        perf_soup = fetch_performance(mid)
        multikills = parse_multikills(perf_soup)

        matches.append({
            "match_id": mid, "region": region,
            "team1": team1, "team2": team2,
            "map_scores": map_scores, "per_map": per_map,
            "aggregate": agg, "multikills": multikills,
        })
        if i < len(all_ids) - 1:
            time.sleep(1.5)
    return matches
