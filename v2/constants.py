"""Shared constants for VFL v2."""

VP_MIN = 5.0
VP_MAX = 15.0
VP_STEP = 0.5
BUDGET = 100
SQUAD_SIZE = 11
TARGET_MEAN = BUDGET / SQUAD_SIZE  # ~9.09

ROLE_SLOTS = {"D": 2, "C": 2, "I": 2, "S": 2}
WILDCARD_SLOTS = 3
MAX_PER_TEAM = 2
MAX_TRANSFERS = 3
NUM_GWS = 6

CHINA_TEAMS = frozenset([
    "Bilibili Gaming", "Dragon Ranger Gaming", "Edward Gaming",
    "FunPlus Phoenix", "JD Mall JDG Esports", "Nova Esports",
    "Titan Esports Club", "Trace Esports", "Wolves Esports",
    "Xi Lai Gaming", "TYLOO", "All Gamers",
])
