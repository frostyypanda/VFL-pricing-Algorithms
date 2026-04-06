"""
VCT 2026 Stage 1 — Complete Schedule Data.

Sourced from vlr.gg (Mar 2026).
Global gameweeks map across all 3 regions.
Americas starts 1 week late (no GW1) and ends 1 week late (only Americas in GW6).
"""

# Gameweek → region mapping
GW_REGIONS = {
    1: ["EMEA", "PAC"],
    2: ["EMEA", "PAC", "AMER"],
    3: ["EMEA", "PAC", "AMER"],
    4: ["EMEA", "PAC", "AMER"],
    5: ["EMEA", "PAC", "AMER"],
    6: ["AMER"],
}

# Team name normalization (user data → VLR names)
TEAM_ALIASES = {
    "ULF Esports": "Eternal Fire",
    "PCIFIC Espor": "PCIFIC Esports",
    "LEVIATÁN": "LEVIATÁN",
    "KRÜ Esports": "KRÜ Esports",
    "LEVIATµN": "LEVIATÁN",
    "KR Esports": "KRÜ Esports",
}

# === GROUP ASSIGNMENTS ===

GROUPS = {
    "AMER": {
        "Alpha": ["G2 Esports", "MIBR", "Cloud9", "LEVIATÁN", "ENVY", "LOUD"],
        "Omega": ["FURIA", "NRG Esports", "100 Thieves", "Evil Geniuses", "Sentinels", "KRÜ Esports"],
    },
    "EMEA": {
        "Alpha": ["Gentle Mates", "Team Heretics", "FUT Esports", "Natus Vincere", "Karmine Corp", "Team Liquid"],
        "Omega": ["BBL Esports", "FNATIC", "Team Vitality", "GIANTX", "Eternal Fire", "PCIFIC Esports"],
    },
    "PAC": {
        "Alpha": ["Nongshim RedForce", "Paper Rex", "DRX", "Global Esports", "Gen.G", "Team Secret"],
        "Omega": ["T1", "Rex Regum Qeon", "Detonation FocusMe", "FULL SENSE", "ZETA DIVISION", "VARREL"],
    },
}

# === PER-GAMEWEEK MATCHUPS ===
# Format: list of (team1, team2) tuples
# Americas regional weeks map to global GWs: AMER W1=GW2, W2=GW3, W3=GW4, W4=GW5, W5=GW6

SCHEDULE = {
    # GW1: EMEA W1 + PAC W1 (no Americas)
    1: {
        "EMEA": [
            ("FNATIC", "Eternal Fire"),
            ("Team Liquid", "Karmine Corp"),
            ("Team Vitality", "GIANTX"),
            ("Gentle Mates", "FUT Esports"),
            ("Team Heretics", "Natus Vincere"),
            ("BBL Esports", "PCIFIC Esports"),
        ],
        "PAC": [
            ("Gen.G", "Global Esports"),
            ("T1", "VARREL"),
            ("Nongshim RedForce", "Paper Rex"),
            ("FULL SENSE", "Detonation FocusMe"),
            ("Team Secret", "DRX"),
            ("Rex Regum Qeon", "ZETA DIVISION"),
        ],
    },
    # GW2: EMEA W2 + PAC W2 + AMER W1
    2: {
        "EMEA": [
            ("PCIFIC Esports", "GIANTX"),
            ("BBL Esports", "Eternal Fire"),
            ("FNATIC", "Team Vitality"),
            ("Team Liquid", "Team Heretics"),
            ("FUT Esports", "Natus Vincere"),
            ("Gentle Mates", "Karmine Corp"),
        ],
        "PAC": [
            ("Detonation FocusMe", "T1"),
            ("Nongshim RedForce", "DRX"),
            ("Team Secret", "Global Esports"),
            ("Rex Regum Qeon", "VARREL"),
            ("FULL SENSE", "ZETA DIVISION"),
            ("Gen.G", "Paper Rex"),
        ],
        "AMER": [
            ("Sentinels", "KRÜ Esports"),
            ("G2 Esports", "MIBR"),
            ("100 Thieves", "Evil Geniuses"),
            ("ENVY", "LOUD"),
            ("Cloud9", "LEVIATÁN"),
            ("FURIA", "NRG Esports"),
        ],
    },
    # GW3: EMEA W3 + PAC W3 + AMER W2
    3: {
        "EMEA": [
            ("BBL Esports", "GIANTX"),
            ("Gentle Mates", "Natus Vincere"),
            ("Eternal Fire", "Team Vitality"),
            ("Karmine Corp", "Team Heretics"),
            ("FUT Esports", "Team Liquid"),
            ("PCIFIC Esports", "FNATIC"),
        ],
        "PAC": [
            ("FULL SENSE", "VARREL"),
            ("Team Secret", "Paper Rex"),
            ("DRX", "Global Esports"),
            ("ZETA DIVISION", "Detonation FocusMe"),
            ("Rex Regum Qeon", "T1"),
            ("Nongshim RedForce", "Gen.G"),
        ],
        "AMER": [
            ("Cloud9", "ENVY"),
            ("FURIA", "Evil Geniuses"),
            ("G2 Esports", "LEVIATÁN"),
            ("MIBR", "LOUD"),
            ("NRG Esports", "KRÜ Esports"),
            ("100 Thieves", "Sentinels"),
        ],
    },
    # GW4: EMEA W4 + PAC W4 + AMER W3
    4: {
        "EMEA": [
            ("Eternal Fire", "PCIFIC Esports"),
            ("Gentle Mates", "Team Heretics"),
            ("GIANTX", "FNATIC"),
            ("Karmine Corp", "FUT Esports"),
            ("Natus Vincere", "Team Liquid"),
            ("BBL Esports", "Team Vitality"),
        ],
        "PAC": [
            ("Rex Regum Qeon", "FULL SENSE"),
            ("Paper Rex", "Global Esports"),
            ("Gen.G", "DRX"),
            ("Detonation FocusMe", "VARREL"),
            ("T1", "ZETA DIVISION"),
            ("Nongshim RedForce", "Team Secret"),
        ],
        "AMER": [
            ("LEVIATÁN", "ENVY"),
            ("G2 Esports", "LOUD"),
            ("FURIA", "KRÜ Esports"),
            ("NRG Esports", "100 Thieves"),
            ("Evil Geniuses", "Sentinels"),
            ("MIBR", "Cloud9"),
        ],
    },
    # GW5: EMEA W5 + PAC W5 + AMER W4
    5: {
        "EMEA": [
            ("Team Heretics", "FUT Esports"),
            ("Team Vitality", "PCIFIC Esports"),
            ("Natus Vincere", "Karmine Corp"),
            ("BBL Esports", "FNATIC"),
            ("GIANTX", "Eternal Fire"),
            ("Gentle Mates", "Team Liquid"),
        ],
        "PAC": [
            ("Gen.G", "Team Secret"),
            ("VARREL", "ZETA DIVISION"),
            ("T1", "FULL SENSE"),
            ("Nongshim RedForce", "Global Esports"),
            ("Paper Rex", "DRX"),
            ("Rex Regum Qeon", "Detonation FocusMe"),
        ],
        "AMER": [
            ("Evil Geniuses", "NRG Esports"),
            ("LEVIATÁN", "MIBR"),
            ("KRÜ Esports", "100 Thieves"),
            ("FURIA", "Sentinels"),
            ("LOUD", "Cloud9"),
            ("G2 Esports", "ENVY"),
        ],
    },
    # GW6: AMER W5 only
    6: {
        "AMER": [
            ("KRÜ Esports", "Evil Geniuses"),
            ("FURIA", "100 Thieves"),
            ("G2 Esports", "Cloud9"),
            ("ENVY", "MIBR"),
            ("LOUD", "LEVIATÁN"),
            ("Sentinels", "NRG Esports"),
        ],
    },
}


def get_team_opponent(team, gw):
    """Get the opponent for a team in a given gameweek. Returns None if team doesn't play."""
    for region, matches in SCHEDULE.get(gw, {}).items():
        for t1, t2 in matches:
            if t1 == team:
                return t2
            if t2 == team:
                return t1
    return None


def get_playing_teams(gw):
    """Get all teams playing in a given gameweek."""
    teams = set()
    for region, matches in SCHEDULE.get(gw, {}).items():
        for t1, t2 in matches:
            teams.add(t1)
            teams.add(t2)
    return teams


def get_team_region(team):
    """Get region for a team."""
    for region, groups in GROUPS.items():
        for group_name, team_list in groups.items():
            if team in team_list:
                return region
    return None
