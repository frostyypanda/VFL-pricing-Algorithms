"""Feature flags for v3 ablation study.

Each flag isolates one of the 5 changes proposed in
output/Stage1_2026_pricing_learnings.md §5.
"""
from dataclasses import dataclass


@dataclass
class V3Flags:
    role_mean_eb: bool = True
    b_floor: bool = True
    team_form_decay: bool = True
    continuity: bool = True
    star_cap: bool = True
    region_quantile: bool = True

    @classmethod
    def all_off(cls):
        return cls(role_mean_eb=False, b_floor=False, team_form_decay=False,
                   continuity=False, star_cap=False, region_quantile=False)

    @classmethod
    def only(cls, name):
        f = cls.all_off()
        setattr(f, name, True)
        return f

    def name(self):
        on = [k for k, v in self.__dict__.items() if v]
        if len(on) == 0:
            return "v3_baseline_v2"
        if len(on) == 6:
            return "v3_full"
        return "v3_" + "_".join(on)
