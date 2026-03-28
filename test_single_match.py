"""
Test: Extract stats from a single VLR match and compute VFL fantasy points.
Validates our parsing logic before scaling up.
"""
import requests
from bs4 import BeautifulSoup
import re, time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def fetch(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_stat_value(cell):
    """Extract the 'both sides' value from a stat cell.
    The first span with class mod-both holds the combined stat."""
    both_span = cell.find("span", class_="mod-both")
    if both_span:
        text = both_span.get_text(strip=True)
        # Remove % signs
        text = text.replace("%", "")
        try:
            return float(text)
        except ValueError:
            return 0
    # Fallback: try the stats-sq span (first number)
    sq = cell.find("span", class_="stats-sq")
    if sq:
        text = sq.get_text(strip=True).replace("%", "")
        try:
            return float(text)
        except ValueError:
            return 0
    return 0


def parse_player_row(tr):
    """Parse a player stats row from a VLR overview table."""
    cells = tr.find_all("td")
    if len(cells) < 10:
        return None

    # Cell 0: player name + team
    player_cell = cells[0]
    if "mod-player" not in (player_cell.get("class") or []):
        return None

    # Extract player name from the link
    player_link = player_cell.find("a")
    if player_link:
        # Player name is in a div inside the link
        name_div = player_link.find("div", class_="text-of")
        if name_div:
            player_name = name_div.get_text(strip=True)
        else:
            player_name = player_link.get_text(strip=True)
    else:
        player_name = player_cell.get_text(strip=True)

    # Extract team name
    team_tag = player_cell.find("span", class_="ge-text-light")
    team_name = team_tag.get_text(strip=True) if team_tag else ""

    # Clean player name (remove team suffix)
    player_name = player_name.replace(team_name, "").strip()

    # Cell 2: Rating, Cell 4: Kills, Cell 5: Deaths, Cell 6: Assists
    rating = parse_stat_value(cells[2])
    kills = parse_stat_value(cells[4])
    deaths = parse_stat_value(cells[5])
    assists = parse_stat_value(cells[6])

    return {
        "player": player_name,
        "team": team_name,
        "rating": rating,
        "kills": int(kills),
        "deaths": int(deaths),
        "assists": int(assists),
    }


def parse_match_overview(soup):
    """Parse per-map stats from a match overview page."""
    # Find map nav items to get map IDs and names
    map_navs = soup.find_all("div", class_="vm-stats-gamesnav-item")
    maps_info = []
    for nav in map_navs:
        game_id = nav.get("data-game-id")
        if game_id and game_id != "all" and nav.get("data-disabled") != "1":
            map_name = nav.get_text(strip=True)
            # Remove leading digit
            map_name = re.sub(r"^\d+", "", map_name).strip()
            maps_info.append({"game_id": game_id, "map_name": map_name})

    # Find map scores from the game headers
    game_divs = soup.find_all("div", class_="vm-stats-game")

    results = {}  # game_id -> {map_name, team1_score, team2_score, players: [...]}

    for mi in maps_info:
        gid = mi["game_id"]
        # Find the vm-stats-game div for this map
        game_div = soup.find("div", class_="vm-stats-game", attrs={"data-game-id": gid})
        if not game_div:
            continue

        # Get map scores from the game header
        header = game_div.find("div", class_="vm-stats-game-header")
        scores = []
        team_names_from_header = []
        if header:
            score_spans = header.find_all("div", class_="score")
            for s in score_spans:
                try:
                    scores.append(int(s.get_text(strip=True)))
                except:
                    pass
            # Get team names from header
            team_divs = header.find_all("div", class_="team-name")
            for td_elem in team_divs:
                team_names_from_header.append(td_elem.get_text(strip=True))

        # Parse player stats from the two tables in this game div
        tables = game_div.find_all("table", class_="wf-table-inset")
        players = []
        team_idx = 0
        for table in tables:
            if "mod-overview" not in (table.get("class") or []):
                continue
            for tr in table.find_all("tr"):
                p = parse_player_row(tr)
                if p:
                    p["team_side"] = team_idx  # 0 = first team, 1 = second team
                    players.append(p)
            team_idx += 1

        results[gid] = {
            "map_name": mi["map_name"],
            "scores": scores,
            "team_names": team_names_from_header,
            "players": players,
        }

    return results


def parse_match_multikills(soup_perf):
    """Parse multi-kill data from performance tab."""
    # Find mod-adv-stats tables
    adv_tables = soup_perf.find_all("table", class_="mod-adv-stats")

    results = []  # list of {player, 2k, 3k, 4k, 5k} per table

    for table in adv_tables:
        # Determine which game_id this table belongs to
        parent_game = table.find_parent("div", class_="vm-stats-game")
        game_id = parent_game.get("data-game-id") if parent_game else "unknown"

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue

            # First cell should be player
            player_cell = cells[0]
            if "mod-player" not in (player_cell.get("class") or []):
                continue

            name_div = player_cell.find("div", class_="text-of")
            player_name = name_div.get_text(strip=True) if name_div else player_cell.get_text(strip=True)
            team_tag = player_cell.find("span", class_="ge-text-light")
            team_name = team_tag.get_text(strip=True) if team_tag else ""
            player_name = player_name.replace(team_name, "").strip()

            # Find the stats - headers are 2K, 3K, 4K, 5K, 1v1, 1v2, ...
            # Cells after player and agents
            stat_cells = [c for c in cells if "mod-stat" in (c.get("class") or [])]

            mk = {"player": player_name, "team": team_name, "game_id": game_id}

            # Get header names from the table
            ths = table.find_all("th")
            header_names = [th.get_text(strip=True) for th in ths]

            for i, sc in enumerate(stat_cells):
                # Map to header (skip first 2 empty headers for player/agent)
                h_idx = i + 2  # offset for player + agent columns
                if h_idx < len(header_names):
                    val_span = sc.find("span", class_="mod-both")
                    if val_span:
                        val = val_span.get_text(strip=True)
                    else:
                        sq = sc.find("span", class_="stats-sq")
                        val = sq.get_text(strip=True) if sq else sc.get_text(strip=True)
                    try:
                        mk[header_names[h_idx]] = int(val)
                    except:
                        mk[header_names[h_idx]] = 0

            results.append(mk)

    return results


def compute_vfl_kill_points(kills_on_map):
    """Compute VFL kill threshold points for a single map."""
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
    """Return which bracket (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50) the kill count falls into."""
    bracket = (kills // 5) * 5
    return min(bracket, 50)


def compute_vfl_map_result_points(team_score, opp_score, won):
    """Compute team-based points for a map result."""
    pts = 0
    if won:
        pts += 1  # map win
        diff = team_score - opp_score
        if diff >= 10:
            pts += 2  # win by 10+
        elif diff >= 5:
            pts += 1  # win by 5+
    else:
        if opp_score - team_score >= 10:
            pts -= 1  # loss by 10+
    return pts


# ── Run test ──
print("Fetching match overview...")
match_url = "https://www.vlr.gg/296735/t1-vs-bleed-champions-tour-2024-pacific-kickoff-group-stage"
soup = fetch(match_url)

# Get overall match result
match_header = soup.find("div", class_="match-header-vs-score")
overall_scores = []
if match_header:
    score_spans = match_header.find_all("span")
    for s in score_spans:
        t = s.get_text(strip=True)
        if t.isdigit():
            overall_scores.append(int(t))
print(f"Overall match score: {overall_scores}")

# Get team names from match header
team_names = []
team_divs = soup.find_all("div", class_="match-header-link-name")
for td in team_divs:
    name = td.find("div", class_="wf-title-med")
    if name:
        team_names.append(name.get_text(strip=True))
print(f"Teams: {team_names}")

# Parse per-map stats
map_stats = parse_match_overview(soup)
print(f"\nFound {len(map_stats)} maps")

for gid, data in map_stats.items():
    print(f"\n{'='*50}")
    print(f"Map: {data['map_name']} | Scores: {data['scores']} | Teams: {data['team_names']}")
    print(f"{'='*50}")
    for p in data["players"]:
        bracket = kill_bracket(p["kills"])
        kill_pts = compute_vfl_kill_points(p["kills"])
        print(f"  {p['team']:12s} {p['player']:15s} K={p['kills']:2d} D={p['deaths']:2d} R={p['rating']:.2f} | kill_bracket={bracket}k kill_pts={kill_pts}")

# Parse multi-kills from performance tab
print("\n\nFetching performance tab...")
time.sleep(1)
perf_url = match_url + "/?game=all&tab=performance"
soup_perf = fetch(perf_url)
mk_data = parse_match_multikills(soup_perf)

print(f"\nMulti-kill data ({len(mk_data)} rows):")
for mk in mk_data:
    fk = mk.get("4K", 0)
    fiv = mk.get("5K", 0)
    if fk or fiv:
        print(f"  {mk['team']:12s} {mk['player']:15s} 4K={fk} 5K={fiv} (game_id={mk['game_id']})")

# Now compute full VFL scoring for this match
print("\n\n" + "=" * 60)
print("VFL SCORING COMPUTATION")
print("=" * 60)

# Determine match winner
if overall_scores and len(overall_scores) >= 2:
    match_winner = team_names[0] if overall_scores[0] > overall_scores[1] else team_names[1]
else:
    match_winner = "?"
print(f"Match winner: {match_winner}")

# Collect all player ratings across the match for TOP1/2/3
all_players_ratings = {}  # player -> overall rating

# Get "all maps" overview for overall ratings
all_div = soup.find("div", class_="vm-stats-game", attrs={"data-game-id": "all"})
if all_div:
    for table in all_div.find_all("table", class_="mod-overview"):
        for tr in table.find_all("tr"):
            p = parse_player_row(tr)
            if p:
                all_players_ratings[p["player"]] = p["rating"]

print(f"\nOverall ratings: {all_players_ratings}")

# Sort by rating for TOP1/2/3
sorted_ratings = sorted(all_players_ratings.items(), key=lambda x: x[1], reverse=True)
top3 = {sorted_ratings[i][0]: 3 - i for i in range(min(3, len(sorted_ratings)))}
print(f"TOP3: {top3}")

# Rating threshold bonuses
rating_bonuses = {}
for player, rating in all_players_ratings.items():
    bonus = 0
    if rating >= 2.0:
        bonus = 3
    elif rating >= 1.75:
        bonus = 2
    elif rating >= 1.5:
        bonus = 1
    rating_bonuses[player] = bonus

# Build multi-kill lookup: need per-map data
# The "all" game_id gives totals, individual game_ids give per-map
mk_by_player_all = {}
for mk in mk_data:
    if mk["game_id"] == "all":
        mk_by_player_all[mk["player"]] = mk

# Compute per-player VFL points
print("\n--- Final VFL Points ---")
num_maps = len(map_stats)

for player_name, overall_rating in all_players_ratings.items():
    # Find per-map kills for this player
    personal_pts = 0
    team_pts = 0
    kill_brackets = {}  # bracket -> count

    for gid, data in map_stats.items():
        for p in data["players"]:
            if p["player"] == player_name:
                # Kill points
                personal_pts += compute_vfl_kill_points(p["kills"])

                # Kill bracket tracking
                b = kill_bracket(p["kills"])
                kill_brackets[b] = kill_brackets.get(b, 0) + 1

                # Team map result points
                scores = data["scores"]
                if len(scores) == 2:
                    if p["team_side"] == 0:
                        won = scores[0] > scores[1]
                        team_pts += compute_vfl_map_result_points(scores[0], scores[1], won)
                    else:
                        won = scores[1] > scores[0]
                        team_pts += compute_vfl_map_result_points(scores[1], scores[0], won)

    # Multi-kill bonuses
    mk = mk_by_player_all.get(player_name, {})
    four_k = mk.get("4K", 0)
    five_k = mk.get("5K", 0)
    personal_pts += four_k * 1 + five_k * 3

    # Rating bonuses
    personal_pts += rating_bonuses.get(player_name, 0)
    personal_pts += top3.get(player_name, 0)

    total = team_pts + personal_pts
    ppm = total / num_maps if num_maps else 0

    print(f"  {player_name:15s} Pts={total:3d} T.Pts={team_pts:2d} P.Pts={personal_pts:2d} PPM={ppm:.2f} | 4K={four_k} 5K={five_k} | brackets={kill_brackets}")
