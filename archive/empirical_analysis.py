"""
Empirical analysis: What best predicts a player's future fantasy performance?

Analyses:
1. Stage-to-stage autocorrelation
2. Recency vs history tradeoff
3. Sample size vs prediction quality (cumulative history)
4. Team strength decomposition
5. Opponent effect (W/L proxy + team strength proxy)
6. Role differences
7. Variance/ceiling analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ── Data loading ──────────────────────────────────────────────────

CHINA_TEAMS = {
    'Bilibili Gaming', 'Dragon Ranger Gaming', 'Edward Gaming',
    'FunPlus Phoenix', 'JD Mall JDG Esports', 'Nova Esports',
    'Titan Esports Club', 'Trace Esports', 'Wolves Esports',
    'Xi Lai Gaming', 'TYLOO', 'All Gamers',
}

STAGE_ORDER = [
    'Kickoff', 'Madrid', 'bangkok', 'Santiago',
    'Stage 1', 'Shanghai', 'Toronto', 'London',
    'Stage 2', 'Champions',
]


def load_data():
    """Load and combine all three years of VFL data."""
    df24 = pd.read_csv('2024 VFL.csv', encoding='latin-1')
    df24.rename(columns={
        df24.columns[0]: 'Team',
        df24.columns[1]: 'Player',
    }, inplace=True)
    df24['Year'] = 2024

    df25 = pd.read_csv('2025 VFL.csv', encoding='latin-1')
    df25['Year'] = 2025

    df26 = pd.read_csv('2026 VFL.csv', encoding='latin-1')
    df26['Year'] = 2026

    df = pd.concat([df24, df25, df26], ignore_index=True)
    df = df[~df['Team'].isin(CHINA_TEAMS)]
    for col in ['P?', 'Pts', 'PPM', 'P.Pts', 'T.Pts']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['P?'] = df['P?'].astype(int)
    return df


def played_only(df):
    """Filter to rows where the player actually played."""
    return df[df['P?'] == 1].copy()


def year_stage_key(year, stage):
    """Create a sortable key for year+stage combinations."""
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    return (year, order.get(stage, 99))


def get_ordered_stages(df):
    """Return list of (year, stage) in chronological order."""
    pairs = df[['Year', 'Stage']].drop_duplicates()
    keyed = [(year_stage_key(y, s), y, s)
             for y, s in zip(pairs['Year'], pairs['Stage'])]
    keyed.sort()
    return [(y, s) for _, y, s in keyed]


# ── Analysis 1: Stage-to-stage autocorrelation ───────────────────

def analysis_1(df):
    """Stage-to-stage autocorrelation of avg Pts."""
    print("=" * 70)
    print("ANALYSIS 1: Stage-to-stage autocorrelation")
    print("=" * 70)

    played = played_only(df)
    stages = get_ordered_stages(played)

    agg = (played.groupby(['Player', 'Year', 'Stage'])
           .agg(avg_pts=('Pts', 'mean'), games=('Pts', 'count'))
           .reset_index())

    results = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        a = agg[(agg['Year'] == y1) & (agg['Stage'] == s1)]
        b = agg[(agg['Year'] == y2) & (agg['Stage'] == s2)]
        merged = a.merge(b, on='Player', suffixes=('_prev', '_next'))
        if len(merged) < 5:
            continue
        r, p = stats.pearsonr(merged['avg_pts_prev'], merged['avg_pts_next'])
        mae = np.mean(np.abs(merged['avg_pts_prev'] - merged['avg_pts_next']))
        is_intl = any(
            x in [s1, s2]
            for x in ['Madrid', 'bangkok', 'Santiago',
                       'Shanghai', 'Toronto', 'London', 'Champions'])
        results.append({
            'from': f"{y1} {s1}", 'to': f"{y2} {s2}",
            'n': len(merged), 'r': r, 'p': p, 'mae': mae,
            'intl': is_intl,
        })
        print(f"  {y1} {s1:12s} -> {y2} {s2:12s}  "
              f"r={r:.3f}  p={p:.4f}  MAE={mae:.2f}  n={len(merged)}")

    domestic = [x for x in results if not x['intl']]
    intl = [x for x in results if x['intl']]
    print(f"\n  Overall median r:           "
          f"{np.median([x['r'] for x in results]):.3f}")
    if domestic:
        print(f"  Domestic-only pairs median r: "
              f"{np.median([x['r'] for x in domestic]):.3f}")
    if intl:
        print(f"  International-involved median r: "
              f"{np.median([x['r'] for x in intl]):.3f}")

    # Also: same-year domestic stage pairs only
    same_year_dom = [x for x in results
                     if not x['intl']
                     and x['from'].split()[0] == x['to'].split()[0]]
    if same_year_dom:
        print(f"  Same-year domestic median r: "
              f"{np.median([x['r'] for x in same_year_dom]):.3f}")
    print()
    return results


# ── Analysis 2: Recency vs history tradeoff ──────────────────────

def blend_metrics(hist, recent, actual, w_old):
    """Compute r and MAE for blended prediction."""
    blend = w_old * hist + (1 - w_old) * recent
    mask = ~(np.isnan(blend) | np.isnan(actual))
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum())
    r, _ = stats.pearsonr(blend[mask], actual[mask])
    mae = float(np.mean(np.abs(blend[mask] - actual[mask])))
    return r, mae, int(mask.sum())


def run_blend_scenario(name, hist_avg, rec_avg, target_avg):
    """Run recency vs history analysis for one scenario."""
    print(f"\n  --- {name} ---")
    combined = pd.concat([hist_avg, rec_avg, target_avg], axis=1)
    combined.columns = ['hist', 'recent', 'actual']

    # History only
    mask_h = combined['hist'].notna() & combined['actual'].notna()
    if mask_h.sum() >= 5:
        r_h, _ = stats.pearsonr(
            combined.loc[mask_h, 'hist'], combined.loc[mask_h, 'actual'])
        mae_h = np.mean(np.abs(
            combined.loc[mask_h, 'hist'] - combined.loc[mask_h, 'actual']))
        print(f"  History only:   r={r_h:.3f}  MAE={mae_h:.2f}  "
              f"n={mask_h.sum()}")

    # Recent only
    mask_r = combined['recent'].notna() & combined['actual'].notna()
    if mask_r.sum() >= 5:
        r_r, _ = stats.pearsonr(
            combined.loc[mask_r, 'recent'], combined.loc[mask_r, 'actual'])
        mae_r = np.mean(np.abs(
            combined.loc[mask_r, 'recent'] - combined.loc[mask_r, 'actual']))
        print(f"  Recent only:    r={r_r:.3f}  MAE={mae_r:.2f}  "
              f"n={mask_r.sum()}")

    # Blends
    mask_b = (combined['hist'].notna() &
              combined['recent'].notna() &
              combined['actual'].notna())
    if mask_b.sum() < 5:
        print(f"  Not enough overlap for blends (n={mask_b.sum()})")
        return
    print(f"\n  Blends (n={mask_b.sum()} players with both):")
    print(f"  {'Weight(old/new)':<20s} {'r':>8s} {'MAE':>8s}")
    best_r, best_w_r = -1, 0
    best_mae, best_w_mae = 999, 0
    for w_old in np.arange(0.0, 1.01, 0.05):
        h = combined.loc[mask_b, 'hist'].values
        rc = combined.loc[mask_b, 'recent'].values
        act = combined.loc[mask_b, 'actual'].values
        r, mae, _ = blend_metrics(h, rc, act, w_old)
        label = f"{w_old:.2f}/{1-w_old:.2f}"
        print(f"  {label:<20s} {r:>8.3f} {mae:>8.2f}")
        if r > best_r:
            best_r, best_w_r = r, w_old
        if mae < best_mae:
            best_mae, best_w_mae = mae, w_old
    print(f"\n  Best r:   w_old={best_w_r:.2f}  r={best_r:.3f}")
    print(f"  Best MAE: w_old={best_w_mae:.2f}  MAE={best_mae:.2f}")


def analysis_2(df):
    """Recency vs history tradeoff."""
    print("=" * 70)
    print("ANALYSIS 2: Recency vs history tradeoff")
    print("=" * 70)
    played = played_only(df)

    # --- Scenario A: Predict 2025 Stage 1 ---
    # History = all 2024, Recent = 2025 Kickoff
    target_a = played[(played['Year'] == 2025) & (played['Stage'] == 'Stage 1')]
    hist_a = played[played['Year'] == 2024]
    rec_a = played[(played['Year'] == 2025) & (played['Stage'] == 'Kickoff')]
    run_blend_scenario(
        'Predicting 2025 Stage 1 (hist=2024, recent=2025 Kickoff)',
        hist_a.groupby('Player')['Pts'].mean(),
        rec_a.groupby('Player')['Pts'].mean(),
        target_a.groupby('Player')['Pts'].mean(),
    )

    # --- Scenario B: Predict 2025 Stage 2 ---
    # History = all 2024, Recent = 2025 pre-Stage2
    target_b = played[(played['Year'] == 2025) & (played['Stage'] == 'Stage 2')]
    hist_b = played[played['Year'] == 2024]
    rec_stages = ['Kickoff', 'bangkok', 'Stage 1', 'Toronto']
    rec_b = played[(played['Year'] == 2025) &
                   (played['Stage'].isin(rec_stages))]
    run_blend_scenario(
        'Predicting 2025 Stage 2 (hist=2024, recent=2025 pre-S2)',
        hist_b.groupby('Player')['Pts'].mean(),
        rec_b.groupby('Player')['Pts'].mean(),
        target_b.groupby('Player')['Pts'].mean(),
    )

    # --- Scenario C: Predict 2025 Stage 1 with history = 2024 full year ---
    # Recent = 2025 Kickoff + Bangkok only
    rec_c = played[(played['Year'] == 2025) &
                   (played['Stage'].isin(['Kickoff', 'bangkok']))]
    run_blend_scenario(
        'Predicting 2025 Stage 1 (hist=2024, recent=2025 KO+Bangkok)',
        hist_a.groupby('Player')['Pts'].mean(),
        rec_c.groupby('Player')['Pts'].mean(),
        target_a.groupby('Player')['Pts'].mean(),
    )

    # --- Scenario D: Predict 2025 Stage 2 with recent = 2025 Stage 1 only
    rec_d = played[(played['Year'] == 2025) & (played['Stage'] == 'Stage 1')]
    run_blend_scenario(
        'Predicting 2025 Stage 2 (hist=2024, recent=2025 Stage 1 only)',
        hist_b.groupby('Player')['Pts'].mean(),
        rec_d.groupby('Player')['Pts'].mean(),
        target_b.groupby('Player')['Pts'].mean(),
    )
    print()


# ── Analysis 3: Sample size vs prediction quality ────────────────

def analysis_3(df):
    """Sample size vs prediction quality using cumulative history."""
    print("=" * 70)
    print("ANALYSIS 3: Sample size vs prediction quality")
    print("=" * 70)

    played = played_only(df)
    stages = get_ordered_stages(played)

    # For each stage, compute cumulative history for each player
    # (all games before that stage), then correlate with that stage
    buckets = {
        '1-3': (1, 3), '4-5': (4, 5), '6-10': (6, 10),
        '11-20': (11, 20), '21-40': (21, 40), '41+': (41, 9999),
    }
    bucket_data = {k: [] for k in buckets}

    for idx in range(1, len(stages)):
        y_tgt, s_tgt = stages[idx]
        target = played[(played['Year'] == y_tgt) &
                        (played['Stage'] == s_tgt)]
        if len(target) < 20:
            continue
        target_avg = target.groupby('Player')['Pts'].mean()

        # Cumulative history = all prior stages
        prior_stages = stages[:idx]
        parts = [played[(played['Year'] == y) & (played['Stage'] == s)]
                 for y, s in prior_stages]
        if not parts:
            continue
        prior = pd.concat(parts)
        prior_agg = prior.groupby('Player').agg(
            hist_avg=('Pts', 'mean'), n_games=('Pts', 'count'))

        merged = prior_agg.join(target_avg.rename('tgt_avg'), how='inner')
        merged = merged.dropna()

        for bname, (lo, hi) in buckets.items():
            sub = merged[(merged['n_games'] >= lo) & (merged['n_games'] <= hi)]
            if len(sub) >= 3:
                bucket_data[bname].append(sub)

    print(f"\n  {'Bucket':<10s} {'n_obs':>8s} {'r':>8s} {'p':>8s} {'MAE':>8s}")
    for bname in buckets:
        dfs = bucket_data[bname]
        if not dfs:
            print(f"  {bname:<10s} {'N/A':>8s}")
            continue
        all_d = pd.concat(dfs)
        if len(all_d) < 5:
            print(f"  {bname:<10s} {len(all_d):>8d} {'N/A':>8s}")
            continue
        r, p = stats.pearsonr(all_d['hist_avg'], all_d['tgt_avg'])
        mae = np.mean(np.abs(all_d['hist_avg'] - all_d['tgt_avg']))
        print(f"  {bname:<10s} {len(all_d):>8d} {r:>8.3f} {p:>8.4f} "
              f"{mae:>8.2f}")
    print()


# ── Analysis 4: Team strength decomposition ─────────────────────

def analysis_4(df):
    """Team strength decomposition."""
    print("=" * 70)
    print("ANALYSIS 4: Team strength decomposition")
    print("=" * 70)

    played = played_only(df)
    total = played['Pts'].sum()
    ppts = played['P.Pts'].sum()
    tpts = played['T.Pts'].sum()
    print(f"\n  Overall: Pts={total:.0f}  "
          f"P.Pts={ppts:.0f} ({100*ppts/total:.1f}%)  "
          f"T.Pts={tpts:.0f} ({100*tpts/total:.1f}%)")

    pa = played.groupby('Player').agg(
        avg_pts=('Pts', 'mean'), avg_ppts=('P.Pts', 'mean'),
        avg_tpts=('T.Pts', 'mean'), games=('Pts', 'count'))
    pa['ppts_frac'] = pa['avg_ppts'] / pa['avg_pts']
    v = pa[pa['avg_pts'] > 0]
    print(f"\n  Per-player P.Pts fraction (n={len(v)}):")
    for lbl, val in [('Mean', v['ppts_frac'].mean()),
                     ('Median', v['ppts_frac'].median()),
                     ('Std', v['ppts_frac'].std()),
                     ('25th', v['ppts_frac'].quantile(0.25)),
                     ('75th', v['ppts_frac'].quantile(0.75))]:
        print(f"    {lbl:<8s}: {val:.3f}")

    # T.Pts stability
    print(f"\n  Team avg T.Pts stability (consecutive stages):")
    stages = get_ordered_stages(played)
    ts = played.groupby(['Team', 'Year', 'Stage'])['T.Pts'].mean().reset_index()
    tpts_corrs = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        a = ts[(ts['Year'] == y1) & (ts['Stage'] == s1)]
        b = ts[(ts['Year'] == y2) & (ts['Stage'] == s2)]
        m = a.merge(b, on='Team', suffixes=('_prev', '_next'))
        if len(m) < 5:
            continue
        r, p = stats.pearsonr(m['T.Pts_prev'], m['T.Pts_next'])
        tpts_corrs.append(r)
        print(f"    {y1} {s1:12s} -> {y2} {s2:12s}  "
              f"r={r:.3f}  n={len(m)}")
    if tpts_corrs:
        print(f"    Median r: {np.median(tpts_corrs):.3f}")

    # Win rate -> future T.Pts
    print(f"\n  Team win rate -> next-stage avg T.Pts:")
    played['win'] = (played['W/L'] == 'W').astype(int)
    wr = (played.groupby(['Team', 'Year', 'Stage'])
          .agg(wr=('win', 'mean'), avg_tpts=('T.Pts', 'mean'))
          .reset_index())
    wr_corrs = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        a = wr[(wr['Year'] == y1) & (wr['Stage'] == s1)]
        b = wr[(wr['Year'] == y2) & (wr['Stage'] == s2)]
        m = a.merge(b, on='Team', suffixes=('_prev', '_next'))
        if len(m) < 5:
            continue
        r, p = stats.pearsonr(m['wr_prev'], m['avg_tpts_next'])
        wr_corrs.append(r)
        print(f"    {y1} {s1:12s} WR -> {y2} {s2:12s} T.Pts  "
              f"r={r:.3f}  n={len(m)}")
    if wr_corrs:
        print(f"    Median r: {np.median(wr_corrs):.3f}")

    # Also: team avg Pts stability (full player performance)
    print(f"\n  Team avg Pts (full) stability:")
    tp = played.groupby(['Team', 'Year', 'Stage'])['Pts'].mean().reset_index()
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        a = tp[(tp['Year'] == y1) & (tp['Stage'] == s1)]
        b = tp[(tp['Year'] == y2) & (tp['Stage'] == s2)]
        m = a.merge(b, on='Team', suffixes=('_prev', '_next'))
        if len(m) < 5:
            continue
        r, _ = stats.pearsonr(m['Pts_prev'], m['Pts_next'])
        print(f"    {y1} {s1:12s} -> {y2} {s2:12s}  r={r:.3f}  n={len(m)}")
    print()


# ── Analysis 5: Opponent / W-L effect ────────────────────────────

def analysis_5(df):
    """Opponent effect via W/L and team strength."""
    print("=" * 70)
    print("ANALYSIS 5: Opponent / Win-Loss effect")
    print("=" * 70)

    played = played_only(df)

    # Part A: How much do W/L outcomes affect scoring?
    print(f"\n  Part A: Performance in Wins vs Losses")
    wins = played[played['W/L'] == 'W']
    losses = played[played['W/L'] == 'L']
    print(f"    {'':15s} {'Wins':>10s} {'Losses':>10s} {'Diff':>10s}")
    for col, lbl in [('Pts', 'Pts'), ('P.Pts', 'P.Pts'), ('T.Pts', 'T.Pts'),
                     ('PPM', 'PPM')]:
        w_m = wins[col].mean()
        l_m = losses[col].mean()
        print(f"    {lbl:<15s} {w_m:>10.2f} {l_m:>10.2f} {w_m-l_m:>+10.2f}")
    print(f"    n(W)={len(wins)}, n(L)={len(losses)}")

    # Part B: Per-player paired W vs L
    print(f"\n  Part B: Per-player paired (same player, W vs L)")
    pw = played.copy()
    pw['is_win'] = (pw['W/L'] == 'W').astype(int)
    ppaired = pw.groupby(['Player', 'is_win'])['Pts'].mean().unstack()
    ppaired.columns = ['loss_avg', 'win_avg']
    ppaired = ppaired.dropna()
    ppaired['diff'] = ppaired['win_avg'] - ppaired['loss_avg']
    print(f"    n players with both W and L: {len(ppaired)}")
    print(f"    Mean Pts in wins:   {ppaired['win_avg'].mean():.2f}")
    print(f"    Mean Pts in losses: {ppaired['loss_avg'].mean():.2f}")
    print(f"    Mean diff:          {ppaired['diff'].mean():+.2f}")
    t, p = stats.ttest_rel(ppaired['win_avg'], ppaired['loss_avg'])
    print(f"    Paired t-test: t={t:.3f}, p={p:.6f}")

    # Part C: Team-strength effect within stages
    print(f"\n  Part C: Player performance by own team strength")
    # For each stage, rank teams by win rate, split into top/bottom half
    stages = get_ordered_stages(played)
    strong_rows = []
    weak_rows = []

    for y, s in stages:
        sd = played[(played['Year'] == y) & (played['Stage'] == s)].copy()
        if len(sd) < 20:
            continue
        sd['win'] = (sd['W/L'] == 'W').astype(int)
        team_wr = sd.groupby('Team')['win'].mean()
        med_wr = team_wr.median()
        strong_teams = set(team_wr[team_wr >= med_wr].index)
        weak_teams = set(team_wr[team_wr < med_wr].index)
        strong_rows.append(sd[sd['Team'].isin(strong_teams)])
        weak_rows.append(sd[sd['Team'].isin(weak_teams)])

    if strong_rows and weak_rows:
        strong = pd.concat(strong_rows)
        weak = pd.concat(weak_rows)
        print(f"    Players on strong teams (above median WR):")
        print(f"      avg Pts={strong['Pts'].mean():.2f}  "
              f"P.Pts={strong['P.Pts'].mean():.2f}  "
              f"T.Pts={strong['T.Pts'].mean():.2f}  n={len(strong)}")
        print(f"    Players on weak teams (below median WR):")
        print(f"      avg Pts={weak['Pts'].mean():.2f}  "
              f"P.Pts={weak['P.Pts'].mean():.2f}  "
              f"T.Pts={weak['T.Pts'].mean():.2f}  n={len(weak)}")
        d_pts = strong['Pts'].mean() - weak['Pts'].mean()
        d_ppts = strong['P.Pts'].mean() - weak['P.Pts'].mean()
        d_tpts = strong['T.Pts'].mean() - weak['T.Pts'].mean()
        print(f"    Difference: Pts={d_pts:+.2f}  "
              f"P.Pts={d_ppts:+.2f}  T.Pts={d_tpts:+.2f}")

    # Part D: Decompose W/L effect
    print(f"\n  Part D: Decomposing the W/L scoring advantage")
    print(f"    T.Pts in wins:    {wins['T.Pts'].mean():.2f}")
    print(f"    T.Pts in losses:  {losses['T.Pts'].mean():.2f}")
    print(f"    P.Pts in wins:    {wins['P.Pts'].mean():.2f}")
    print(f"    P.Pts in losses:  {losses['P.Pts'].mean():.2f}")
    total_diff = wins['Pts'].mean() - losses['Pts'].mean()
    tpts_diff = wins['T.Pts'].mean() - losses['T.Pts'].mean()
    ppts_diff = wins['P.Pts'].mean() - losses['P.Pts'].mean()
    print(f"    Total diff:  {total_diff:+.2f}")
    print(f"    From T.Pts:  {tpts_diff:+.2f} "
          f"({100*tpts_diff/total_diff:.1f}%)")
    print(f"    From P.Pts:  {ppts_diff:+.2f} "
          f"({100*ppts_diff/total_diff:.1f}%)")
    print()


# ── Analysis 6: Role differences ────────────────────────────────

def analysis_6(df):
    """Role-based analysis with data-driven thresholds."""
    print("=" * 70)
    print("ANALYSIS 6: Role differences")
    print("=" * 70)

    played = played_only(df)

    pa = played.groupby('Player').agg(
        total_pts=('Pts', 'sum'), total_ppts=('P.Pts', 'sum'),
        avg_pts=('Pts', 'mean'), std_pts=('Pts', 'std'),
        games=('Pts', 'count'))
    pa = pa[pa['total_pts'] > 0]
    pa['ppts_ratio'] = pa['total_ppts'] / pa['total_pts']

    # Show the distribution of P.Pts/Pts ratio
    print(f"\n  P.Pts/Pts ratio distribution (n={len(pa)}):")
    for q in [0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9]:
        print(f"    {q*100:.0f}th pctl: {pa['ppts_ratio'].quantile(q):.3f}")

    # Use quartile-based role classification
    q25 = pa['ppts_ratio'].quantile(0.25)
    q50 = pa['ppts_ratio'].quantile(0.50)
    q75 = pa['ppts_ratio'].quantile(0.75)
    print(f"\n  Quartile thresholds: Q25={q25:.3f}, Q50={q50:.3f}, "
          f"Q75={q75:.3f}")

    def classify(ratio):
        if ratio >= q75:
            return 'High P.Pts (Duelist-like)'
        elif ratio >= q50:
            return 'Med-High P.Pts (Initiator-like)'
        elif ratio >= q25:
            return 'Med-Low P.Pts (Controller-like)'
        else:
            return 'Low P.Pts (Sentinel-like)'

    pa['role'] = pa['ppts_ratio'].apply(classify)
    pa['cv'] = pa['std_pts'] / pa['avg_pts']

    roles_ordered = [
        'High P.Pts (Duelist-like)',
        'Med-High P.Pts (Initiator-like)',
        'Med-Low P.Pts (Controller-like)',
        'Low P.Pts (Sentinel-like)',
    ]

    print(f"\n  Role characteristics:")
    print(f"  {'Role':<35s} {'n':>5s} {'AvgPts':>8s} {'Std':>8s} "
          f"{'CV':>8s} {'PPtsR':>8s}")
    for role in roles_ordered:
        sub = pa[pa['role'] == role]
        print(f"  {role:<35s} {len(sub):>5d} "
              f"{sub['avg_pts'].mean():>8.2f} "
              f"{sub['std_pts'].mean():>8.2f} "
              f"{sub['cv'].mean():>8.3f} "
              f"{sub['ppts_ratio'].mean():>8.3f}")

    # Prediction quality by role
    stages = get_ordered_stages(played)
    role_data = {r: [] for r in roles_ordered}
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        prev = played[(played['Year'] == y1) & (played['Stage'] == s1)]
        nxt = played[(played['Year'] == y2) & (played['Stage'] == s2)]
        prev_a = prev.groupby('Player')['Pts'].mean().rename('prev')
        nxt_a = nxt.groupby('Player')['Pts'].mean().rename('next')
        m = pd.concat([prev_a, nxt_a], axis=1).dropna()
        m = m.join(pa[['role']]).dropna(subset=['role'])
        for role in roles_ordered:
            rs = m[m['role'] == role]
            if len(rs) >= 5:
                role_data[role].append(rs)

    print(f"\n  Stage-to-stage prediction by role:")
    print(f"  {'Role':<35s} {'n':>6s} {'r':>8s} {'MAE':>8s}")
    for role in roles_ordered:
        if not role_data[role]:
            print(f"  {role:<35s} {'N/A':>6s}")
            continue
        all_d = pd.concat(role_data[role])
        r, p = stats.pearsonr(all_d['prev'], all_d['next'])
        mae = np.mean(np.abs(all_d['prev'] - all_d['next']))
        print(f"  {role:<35s} {len(all_d):>6d} {r:>8.3f} {mae:>8.2f}")

    # Also: does the P.Pts/Pts ratio itself predict future performance?
    print(f"\n  Does P.Pts ratio predict next-stage avg Pts?")
    ratio_pred = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        prev = played[(played['Year'] == y1) & (played['Stage'] == s1)]
        nxt = played[(played['Year'] == y2) & (played['Stage'] == s2)]
        prev_a = prev.groupby('Player').agg(
            avg_pts=('Pts', 'mean'), ppts_r=('P.Pts', 'sum'))
        prev_pts_sum = prev.groupby('Player')['Pts'].sum()
        prev_a['ppts_ratio'] = prev_a['ppts_r'] / prev_pts_sum
        prev_a = prev_a[prev_pts_sum > 0]
        nxt_a = nxt.groupby('Player')['Pts'].mean().rename('next_avg')
        m = prev_a.join(nxt_a, how='inner').dropna()
        if len(m) >= 5:
            ratio_pred.append(m)
    if ratio_pred:
        all_rp = pd.concat(ratio_pred)
        r_ratio, p_ratio = stats.pearsonr(
            all_rp['ppts_ratio'], all_rp['next_avg'])
        r_avg, p_avg = stats.pearsonr(
            all_rp['avg_pts'], all_rp['next_avg'])
        print(f"    P.Pts ratio -> next avg: r={r_ratio:.3f}  "
              f"p={p_ratio:.4f}  n={len(all_rp)}")
        print(f"    Avg Pts -> next avg:     r={r_avg:.3f}  "
              f"p={p_avg:.4f}  (baseline)")
    print()


# ── Analysis 7: Variance/ceiling analysis ────────────────────────

def analysis_7(df):
    """Variance and ceiling analysis."""
    print("=" * 70)
    print("ANALYSIS 7: Variance/ceiling analysis")
    print("=" * 70)

    played = played_only(df)

    pa = played.groupby('Player').agg(
        avg_pts=('Pts', 'mean'), std_pts=('Pts', 'std'),
        p90=('Pts', lambda x: np.percentile(x, 90)),
        p10=('Pts', lambda x: np.percentile(x, 10)),
        max_pts=('Pts', 'max'), min_pts=('Pts', 'min'),
        games=('Pts', 'count'))
    pa = pa[pa['games'] >= 5]
    pa['cv'] = pa['std_pts'] / pa['avg_pts']
    pa['p90_uplift'] = pa['p90'] - pa['avg_pts']
    pa['range'] = pa['max_pts'] - pa['min_pts']

    print(f"\n  Overall stats (n={len(pa)}, min 5 games):")
    for lbl, col in [('Avg Pts', 'avg_pts'), ('Std Pts', 'std_pts'),
                     ('CV', 'cv'), ('P90', 'p90'), ('P90 uplift', 'p90_uplift'),
                     ('Max', 'max_pts'), ('Range', 'range')]:
        print(f"    {lbl:<15s}: mean={pa[col].mean():.2f}  "
              f"median={pa[col].median():.2f}")

    # Variance stability
    print(f"\n  Variance (std) stability across consecutive stages:")
    stages = get_ordered_stages(played)
    var_corrs = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        prev = played[(played['Year'] == y1) & (played['Stage'] == s1)]
        nxt = played[(played['Year'] == y2) & (played['Stage'] == s2)]
        pa_p = prev.groupby('Player').agg(
            std_p=('Pts', 'std'), n_p=('Pts', 'count'))
        pa_n = nxt.groupby('Player').agg(
            std_n=('Pts', 'std'), n_n=('Pts', 'count'))
        m = pa_p.join(pa_n, how='inner').dropna()
        m = m[(m['n_p'] >= 3) & (m['n_n'] >= 3)]
        if len(m) < 5:
            continue
        r, p = stats.pearsonr(m['std_p'], m['std_n'])
        var_corrs.append(r)
        print(f"    {y1} {s1:12s} -> {y2} {s2:12s}  "
              f"r(std)={r:.3f}  n={len(m)}")
    if var_corrs:
        print(f"    Median r(std): {np.median(var_corrs):.3f}")

    # CV stability
    print(f"\n  CV stability across consecutive stages:")
    cv_corrs = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        prev = played[(played['Year'] == y1) & (played['Stage'] == s1)]
        nxt = played[(played['Year'] == y2) & (played['Stage'] == s2)]
        pa_p = prev.groupby('Player').agg(
            avg_p=('Pts', 'mean'), std_p=('Pts', 'std'),
            n_p=('Pts', 'count'))
        pa_n = nxt.groupby('Player').agg(
            avg_n=('Pts', 'mean'), std_n=('Pts', 'std'),
            n_n=('Pts', 'count'))
        m = pa_p.join(pa_n, how='inner').dropna()
        m = m[(m['n_p'] >= 3) & (m['n_n'] >= 3)]
        m = m[(m['avg_p'] > 0) & (m['avg_n'] > 0)]
        if len(m) < 5:
            continue
        m['cv_p'] = m['std_p'] / m['avg_p']
        m['cv_n'] = m['std_n'] / m['avg_n']
        r, p = stats.pearsonr(m['cv_p'], m['cv_n'])
        cv_corrs.append(r)
        print(f"    {y1} {s1:12s} -> {y2} {s2:12s}  "
              f"r(CV)={r:.3f}  n={len(m)}")
    if cv_corrs:
        print(f"    Median r(CV): {np.median(cv_corrs):.3f}")

    # High-variance vs low-variance: who outperforms?
    print(f"\n  High-variance vs low-variance: regression to mean")
    op_data = []
    for i in range(len(stages) - 1):
        y1, s1 = stages[i]
        y2, s2 = stages[i + 1]
        prev = played[(played['Year'] == y1) & (played['Stage'] == s1)]
        nxt = played[(played['Year'] == y2) & (played['Stage'] == s2)]
        pa_p = prev.groupby('Player').agg(
            avg_p=('Pts', 'mean'), std_p=('Pts', 'std'),
            n_p=('Pts', 'count'))
        pa_n = nxt.groupby('Player')['Pts'].mean().rename('avg_n')
        m = pa_p.join(pa_n, how='inner').dropna()
        m = m[m['n_p'] >= 3]
        if len(m) >= 5:
            m['cv_p'] = m['std_p'] / m['avg_p'].replace(0, np.nan)
            m['resid'] = m['avg_n'] - m['avg_p']
            op_data.append(m)

    if op_data:
        all_op = pd.concat(op_data).replace([np.inf, -np.inf], np.nan).dropna()
        cv_med = all_op['cv_p'].median()
        lo_cv = all_op[all_op['cv_p'] < cv_med]
        hi_cv = all_op[all_op['cv_p'] >= cv_med]
        print(f"    CV median: {cv_med:.3f}")
        print(f"    Low-CV (n={len(lo_cv)}): "
              f"mean prev={lo_cv['avg_p'].mean():.2f}  "
              f"mean next={lo_cv['avg_n'].mean():.2f}  "
              f"resid={lo_cv['resid'].mean():+.2f}")
        print(f"    High-CV (n={len(hi_cv)}): "
              f"mean prev={hi_cv['avg_p'].mean():.2f}  "
              f"mean next={hi_cv['avg_n'].mean():.2f}  "
              f"resid={hi_cv['resid'].mean():+.2f}")
        r_cv_r, p_cv_r = stats.pearsonr(all_op['cv_p'], all_op['resid'])
        r_cv_n, p_cv_n = stats.pearsonr(all_op['cv_p'], all_op['avg_n'])
        print(f"    CV -> residual:  r={r_cv_r:.3f}  p={p_cv_r:.4f}")
        print(f"    CV -> next avg:  r={r_cv_n:.3f}  p={p_cv_n:.4f}")

    # Ceiling by tier
    print(f"\n  P90 uplift by player tier:")
    pa['tier'] = pd.qcut(pa['avg_pts'], q=4,
                         labels=['Bottom 25%', '25-50%', '50-75%', 'Top 25%'])
    print(f"  {'Tier':<12s} {'n':>5s} {'Avg':>7s} {'P10':>7s} {'P90':>7s} "
          f"{'Uplift':>8s} {'Std':>7s} {'CV':>7s}")
    for tier in ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']:
        sub = pa[pa['tier'] == tier]
        print(f"  {tier:<12s} {len(sub):>5d} "
              f"{sub['avg_pts'].mean():>7.2f} "
              f"{sub['p10'].mean():>7.2f} "
              f"{sub['p90'].mean():>7.2f} "
              f"{sub['p90_uplift'].mean():>8.2f} "
              f"{sub['std_pts'].mean():>7.2f} "
              f"{sub['cv'].mean():>7.3f}")

    # Top 20 highest-ceiling players
    print(f"\n  Top 20 players by P90:")
    top20 = pa.nlargest(20, 'p90')
    print(f"  {'Player':<20s} {'Avg':>7s} {'P90':>7s} {'Max':>7s} "
          f"{'Std':>7s} {'Games':>6s}")
    for _, row in top20.iterrows():
        print(f"  {row.name:<20s} {row['avg_pts']:>7.2f} {row['p90']:>7.2f} "
              f"{row['max_pts']:>7.0f} {row['std_pts']:>7.2f} "
              f"{row['games']:>6.0f}")
    print()


# ── Summary ──────────────────────────────────────────────────────

def summary(df):
    """Print key takeaways."""
    print("=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    print("""
  1. AUTOCORRELATION: Stage-to-stage r ~ 0.3-0.6. Performance IS
     somewhat predictable but with substantial noise. International
     events are noisier.

  2. RECENCY: The optimal blend depends on how much recent data exists.
     With only Kickoff data (small sample), history gets MORE weight.
     With a full stage of recent data, recent gets MORE weight.
     Practical: ~70% old / 30% recent when recent = Kickoff only;
     ~100% recent when recent = full stage + internationals.

  3. SAMPLE SIZE: Prediction improves dramatically from 1-3 to 6-10
     games. Beyond ~10 games, marginal improvement is small.
     Bayesian shrinkage toward population mean for <6 games is justified.

  4. TEAM STRENGTH: T.Pts = ~32% of total scoring. Team T.Pts is
     POORLY predictable across stages (median r < 0.2). Win rate is
     a weak predictor of future T.Pts. Team effects are real but
     unstable -- use with caution.

  5. OPPONENT: Players score ~5.8 more in wins vs losses. ~55% of this
     is T.Pts (map win bonuses) and ~45% is P.Pts (kill more when
     winning). Being on a strong team adds ~2-3 pts/game total.

  6. ROLES: High P.Pts ratio players (duelist-like) have higher
     variance (CV). Role type moderately affects prediction quality.
     The P.Pts ratio itself has weak predictive power for absolute
     performance.

  7. VARIANCE: Player variance is NOT stable across stages (median
     r(std) ~ 0.08). High-CV players don't reliably outperform their
     mean. Variance-based pricing adjustments should be conservative.
     P90 uplift scales with player tier (~4 for bottom, ~5.5 for top).
""")


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("\nLoading data...")
    df = load_data()
    played = played_only(df)
    print(f"  Total rows: {len(df)}")
    print(f"  Played rows (P?=1): {len(played)}")
    print(f"  Unique players: {played['Player'].nunique()}")
    print(f"  Unique teams: {played['Team'].nunique()}")
    print(f"  Years: {sorted(played['Year'].unique())}")
    print()

    analysis_1(df)
    analysis_2(df)
    analysis_3(df)
    analysis_4(df)
    analysis_5(df)
    analysis_6(df)
    analysis_7(df)
    summary(df)

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
