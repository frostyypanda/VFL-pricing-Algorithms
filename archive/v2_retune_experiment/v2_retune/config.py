"""Hyperparameter search space for v2 retune."""
from dataclasses import dataclass, field
from typing import List, Tuple

CAL_WINDOWS = {
    "v2_default": [("Stage 1", 2025), ("Stage 2", 2025)],
    "plus_2026K": [("Stage 1", 2025), ("Stage 2", 2025), ("Kickoff", 2026)],
    "plus_2026KS": [("Stage 1", 2025), ("Stage 2", 2025),
                     ("Kickoff", 2026), ("Santiago", 2026)],
    "recent_only": [("Stage 2", 2025), ("Kickoff", 2026), ("Santiago", 2026)],
}

RECENCY_MAPS = {
    "v2_default": {
        "Kickoff": 0, "bangkok": 1, "Madrid": 1, "Santiago": 1,
        "Stage 1": 2, "Toronto": 3, "Shanghai": 3, "London": 3,
        "Stage 2": 4, "Champions": 5,
    },
    "strict": {
        "Kickoff": 0, "bangkok": 1, "Madrid": 2, "Santiago": 3,
        "Stage 1": 4, "Toronto": 5, "Shanghai": 6, "London": 7,
        "Stage 2": 8, "Champions": 9,
    },
}

# Ensemble weight simplex (w_eb, w_ridge, w_ema), each ≥0, sum=1
ENSEMBLE_WEIGHTS = []
for w_eb in [0.0, 0.2, 0.33, 0.4, 0.6, 0.8, 1.0]:
    for w_ridge in [0.0, 0.2, 0.33, 0.4, 0.6, 0.8]:
        w_ema = 1.0 - w_eb - w_ridge
        if w_ema < -1e-9 or w_ema > 1 + 1e-9:
            continue
        if w_ema < 0:
            continue
        ENSEMBLE_WEIGHTS.append((round(w_eb, 2), round(w_ridge, 2),
                                  round(w_ema, 2)))

PICKRATE_WEIGHTS = [0.0, 0.05]

HOLDOUTS = [
    (2024, "Stage 1"), (2024, "Stage 2"),
    (2025, "Kickoff"), (2025, "Stage 1"), (2025, "Stage 2"),
    (2026, "Kickoff"), (2026, "Santiago"), (2026, "Stage 1"),
]


@dataclass
class TuneConfig:
    cal_name: str
    recency_name: str
    ensemble: Tuple[float, float, float]
    pickrate_w: float

    def label(self):
        e = "_".join(f"{x:.2f}" for x in self.ensemble)
        return f"{self.cal_name}|{self.recency_name}|{e}|pr{self.pickrate_w}"
