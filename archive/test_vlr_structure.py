"""
Test script to probe VLR.gg HTML structure for a single match.
We'll inspect the actual class names and element hierarchy.
"""
import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

# ── Test 1: Match overview page structure ──
print("=" * 60)
print("TEST 1: Match overview page (T1 vs BLEED)")
print("=" * 60)
match_url = "https://www.vlr.gg/296735/t1-vs-bleed-champions-tour-2024-pacific-kickoff-group-stage"
soup = fetch(match_url)

# Find all divs with 'vm-stats-game' in class
game_divs = soup.find_all("div", class_=lambda c: c and "vm-stats-game" in c)
print(f"\nFound {len(game_divs)} vm-stats-game divs")
for i, gd in enumerate(game_divs):
    classes = gd.get("class", [])
    data_attrs = {k: v for k, v in gd.attrs.items() if k.startswith("data-")}
    print(f"  Game div {i}: classes={classes}, data={data_attrs}")

# Look at map headers / scores
print("\n--- Map headers ---")
map_headers = soup.find_all("div", class_=lambda c: c and "map" in str(c).lower())
for mh in map_headers[:10]:
    cls = mh.get("class", [])
    text = mh.get_text(strip=True)[:100]
    print(f"  class={cls} text='{text}'")

# Look for score elements
print("\n--- Score elements ---")
score_els = soup.find_all("div", class_=lambda c: c and "score" in str(c).lower())
for se in score_els[:10]:
    cls = se.get("class", [])
    text = se.get_text(strip=True)[:80]
    print(f"  class={cls} text='{text}'")

# Look at stats tables
print("\n--- Stats tables ---")
tables = soup.find_all("table")
print(f"Found {len(tables)} tables")
for i, t in enumerate(tables[:5]):
    classes = t.get("class", [])
    rows = t.find_all("tr")
    print(f"  Table {i}: classes={classes}, rows={len(rows)}")
    # Print first row header
    if rows:
        ths = rows[0].find_all("th")
        if ths:
            print(f"    Headers: {[th.get_text(strip=True) for th in ths]}")
        tds = rows[0].find_all("td")
        if tds:
            first_row_text = [td.get_text(strip=True)[:20] for td in tds]
            print(f"    First row cells: {first_row_text}")

# Look at player stat rows more carefully
print("\n--- Player rows in first table ---")
if tables:
    for tr in tables[0].find_all("tr")[:3]:
        cells = tr.find_all(["td", "th"])
        for ci, cell in enumerate(cells):
            cls = cell.get("class", [])
            text = cell.get_text(strip=True)[:30]
            spans = cell.find_all("span")
            span_classes = [s.get("class", []) for s in spans]
            print(f"    Cell {ci}: class={cls} text='{text}' spans={span_classes}")
        print("    ---")

# ── Test 2: Match performance page (multi-kills) ──
print("\n" + "=" * 60)
print("TEST 2: Performance tab (multi-kills)")
print("=" * 60)
time.sleep(1)
perf_url = match_url + "/?game=all&tab=performance"
soup2 = fetch(perf_url)

# Look for performance-specific elements
print("\n--- Performance containers ---")
perf_divs = soup2.find_all("div", class_=lambda c: c and "performance" in str(c).lower())
print(f"Found {len(perf_divs)} performance divs")

# Look for multi-kill related text
print("\n--- Elements containing '2k' or '4k' text ---")
for el in soup2.find_all(string=lambda s: s and ("2k" in s.lower() or "4k" in s.lower() or "5k" in s.lower())):
    parent = el.parent
    if parent:
        pcls = parent.get("class", [])
        print(f"  parent tag={parent.name} class={pcls} text='{el.strip()[:50]}'")

# All tables on performance page
print("\n--- Performance page tables ---")
tables2 = soup2.find_all("table")
print(f"Found {len(tables2)} tables on performance page")
for i, t in enumerate(tables2[:5]):
    classes = t.get("class", [])
    rows = t.find_all("tr")
    print(f"  Table {i}: classes={classes}, rows={len(rows)}")
    if rows:
        ths = rows[0].find_all("th")
        if ths:
            print(f"    Headers: {[th.get_text(strip=True) for th in ths]}")

# ── Test 3: Event matches page ──
print("\n" + "=" * 60)
print("TEST 3: Event matches page")
print("=" * 60)
time.sleep(1)
event_url = "https://www.vlr.gg/event/matches/1924/champions-tour-2024-pacific-kickoff/?series_id=all"
soup3 = fetch(event_url)

# Find match links
print("\n--- Match links ---")
match_links = soup3.find_all("a", href=lambda h: h and h.startswith("/") and len(h.split("/")) >= 3)
match_ids = set()
for a in match_links:
    href = a.get("href", "")
    parts = href.strip("/").split("/")
    if parts and parts[0].isdigit() and "vs" in href:
        match_ids.add(parts[0])

print(f"Found {len(match_ids)} unique match IDs: {sorted(match_ids)}")

# Look at match card structure
print("\n--- Match card elements ---")
match_cards = soup3.find_all("a", class_=lambda c: c and "match" in str(c).lower())
print(f"Found {len(match_cards)} match card elements")
if match_cards:
    mc = match_cards[0]
    print(f"  First card class: {mc.get('class')}")
    print(f"  href: {mc.get('href')}")
    # Show children structure
    for child in mc.children:
        if hasattr(child, 'get'):
            print(f"    child tag={child.name} class={child.get('class', [])}")

print("\nDone!")
