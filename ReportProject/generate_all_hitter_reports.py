import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import re

# ── SETTINGS ──────────────────────────────────────────────────────────────────
import os as _os
_SCRIPT_DIR  = _os.path.dirname(_os.path.abspath(__file__))

CSV_FILE     = _os.path.join(_SCRIPT_DIR, 'NDSU_Hitting.csv')
OUTPUT_DIR   = _os.path.join(_SCRIPT_DIR, 'batter_reports')

SCHOOL_COLOR = "#FDB719"
BG_DARK      = '#1a1a2e'
BG_LIGHT     = '#f5f5f5'

hit_type_styles = {
    'Single':        {'color': '#2ecc71', 'marker': 'o'},
    'Double':        {'color': '#3498db', 'marker': 'o'},
    'Triple':        {'color': '#9b59b6', 'marker': 'o'},
    'HomeRun':       {'color': '#e74c3c', 'marker': '*'},
    'Out':           {'color': '#95a5a6', 'marker': 'x'},
    'FieldersChoice':{'color': '#f39c12', 'marker': 'o'},
    'Error':         {'color': '#e67e22', 'marker': 'D'},
}

pitch_type_styles = {
    'Fastball':  '#e74c3c',
    'Slider':    '#C9A227',
    'Sinker':    '#e67e22',
    'Cutter':    '#9b59b6',
    'Curveball': '#3498db',
    'Changeup':  '#2ecc71',
    'Splitter':  '#f39c12',
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: resolve pitch type column consistently
# ══════════════════════════════════════════════════════════════════════════════
def resolve_pitch_type(df):
    """
    Return a Series using TaggedPitchType where available,
    falling back to AutoPitchType. Still-unresolved rows become 'Unknown'.
    """
    tagged = df['taggedpitchtype'].astype(str).str.strip()
    auto   = (df['autopitchtype'].astype(str).str.strip()
              if 'autopitchtype' in df.columns
              else pd.Series('', index=df.index))

    result = tagged.copy()
    fallback_mask = result.isin(['', 'nan', 'Undefined', 'Unknown', 'None'])
    result[fallback_mask] = auto[fallback_mask]
    result[result.isin(['', 'nan', 'Undefined', 'Unknown', 'None'])] = 'Unknown'
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: draw strike zone
# ══════════════════════════════════════════════════════════════════════════════
def draw_strike_zone(ax):
    zone_left, zone_right = -0.708, 0.708
    sz_bot, sz_top = 1.5, 3.5
    zone = plt.Rectangle((zone_left, sz_bot),
                          zone_right - zone_left, sz_top - sz_bot,
                          linewidth=2, edgecolor='black', facecolor='none', zorder=5)
    ax.add_patch(zone)
    for xl in [-0.236, 0.236]:
        ax.plot([xl, xl], [sz_bot, sz_top], 'k-', lw=0.8, alpha=0.5)
    for yl in [2.167, 2.833]:
        ax.plot([zone_left, zone_right], [yl, yl], 'k-', lw=0.8, alpha=0.5)
    plate_x = [-0.708, 0.708, 0.708,  0,    -0.708, -0.708]
    plate_y = [0.5,    0.5,   0.35,   0.1,   0.35,   0.5]
    ax.plot(plate_x, plate_y, 'k-', lw=1.5)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE ONE REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(df, batter_name, output_path):

    # Attach a consistent pitch type column used by BOTH the bar chart and table
    df = df.copy()
    df['_pitch_type'] = resolve_pitch_type(df)

    # ── STATS ────────────────────────────────────────────────────────────────
    pa  = (df['korbb'].isin(['Strikeout', 'Walk']) |
           df['playresult'].isin(['Single','Double','Triple','HomeRun',
                                  'Out','Error','FieldersChoice','Sacrifice']) |
           df['pitchcall'].eq('HitByPitch')).sum()

    ab  = (df['korbb'].eq('Strikeout') |
           df['playresult'].isin(['Single','Double','Triple','HomeRun',
                                  'Out','Error','FieldersChoice'])).sum()

    h   = df['playresult'].isin(['Single','Double','Triple','HomeRun']).sum()
    dbl = df['playresult'].eq('Double').sum()
    tpl = df['playresult'].eq('Triple').sum()
    hr  = df['playresult'].eq('HomeRun').sum()
    bb  = df['korbb'].eq('Walk').sum()
    k   = df['korbb'].eq('Strikeout').sum()
    hbp = df['pitchcall'].eq('HitByPitch').sum()

    avg = h / ab                          if ab  else 0.0
    obp = (h + bb + hbp) / pa            if pa  else 0.0
    slg = (h + dbl + 2*tpl + 3*hr) / ab  if ab  else 0.0
    ops = obp + slg

    swings_mask = df['pitchcall'].isin(['StrikeSwinging','FoulBallNotFieldable','FoulBall','InPlay'])
    swings_n    = swings_mask.sum()
    misses      = df['pitchcall'].eq('StrikeSwinging').sum()
    whiff_pct   = misses / swings_n * 100 if swings_n else 0.0
    bb_pct      = bb / pa * 100           if pa       else 0.0
    k_pct       = k  / pa * 100           if pa       else 0.0

    inplay = df[df['pitchcall'] == 'InPlay']
    ev     = pd.to_numeric(inplay['exitspeed'], errors='coerce').dropna()
    avg_ev = ev.mean() if len(ev) else None
    max_ev = ev.max()  if len(ev) else None

    # ── FIGURE ───────────────────────────────────────────────────────────────
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor(BG_LIGHT)

        gs = gridspec.GridSpec(
            5, 2,
            figure=fig,
            height_ratios=[0.12, 0.18, 0.38, 0.28, 0.4],
            hspace=0.25,
            wspace=0.35,
            left=0.08, right=0.95,
            top=0.96, bottom=0.04
        )

        # ── HEADER ───────────────────────────────────────────────────────────
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.set_facecolor(SCHOOL_COLOR)
        for spine in ax_header.spines.values():
            spine.set_visible(False)
        ax_header.set_xticks([]); ax_header.set_yticks([])
        ax_header.text(0.5, 0.65, batter_name,
                       transform=ax_header.transAxes,
                       fontsize=18, fontweight='bold', color='black',
                       ha='center', va='center')
        ax_header.text(0.5, 0.2, 'Hitting Report  |  Series Against NDSU',
                       transform=ax_header.transAxes,
                       fontsize=9, color='black', alpha=0.85,
                       ha='center', va='center')

        # ── STAT TABLE ───────────────────────────────────────────────────────
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        col_labels = ['PA', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'K',
                      'AVG', 'OBP', 'SLG', 'OPS', 'BB%', 'K%', 'Whiff%',
                      'Avg EV', 'Max EV']
        col_values = [
            str(pa), str(ab), str(h), str(dbl), str(tpl), str(hr),
            str(bb), str(k),
            f'{avg:.3f}', f'{obp:.3f}', f'{slg:.3f}', f'{ops:.3f}',
            f'{bb_pct:.1f}%', f'{k_pct:.1f}%', f'{whiff_pct:.1f}%',
            f'{avg_ev:.1f}' if avg_ev is not None else 'N/A',
            f'{max_ev:.1f}' if max_ev is not None else 'N/A',
        ]

        n = len(col_labels)
        xs = np.linspace(0.01, 0.99, n + 1)
        cell_w = xs[1] - xs[0]

        for i, (lbl, val) in enumerate(zip(col_labels, col_values)):
            cx = xs[i] + cell_w / 2
            rect = FancyBboxPatch((xs[i] + 0.002, 0.52), cell_w - 0.004, 0.38,
                                   boxstyle='round,pad=0.01',
                                   facecolor=SCHOOL_COLOR, edgecolor='white',
                                   linewidth=0.5,
                                   transform=ax_table.transAxes, clip_on=False)
            ax_table.add_patch(rect)
            ax_table.text(cx, 0.71, lbl,
                          transform=ax_table.transAxes,
                          fontsize=7.5, fontweight='bold', color='black',
                          ha='center', va='center')
            rect2 = FancyBboxPatch((xs[i] + 0.002, 0.10), cell_w - 0.004, 0.38,
                                    boxstyle='round,pad=0.01',
                                    facecolor='white', edgecolor='#cccccc',
                                    linewidth=0.5,
                                    transform=ax_table.transAxes, clip_on=False)
            ax_table.add_patch(rect2)
            ax_table.text(cx, 0.29, val,
                          transform=ax_table.transAxes,
                          fontsize=8, color='#222222',
                          ha='center', va='center')

        # ── SPRAY CHART ──────────────────────────────────────────────────────
        ax_spray = fig.add_subplot(gs[2, 0])
        ax_spray.set_facecolor('#3a7d44')

        # Only keep InPlay rows with valid direction & distance
        hits_df = df[df['pitchcall'] == 'InPlay'].copy()
        hits_df['direction'] = pd.to_numeric(hits_df['direction'], errors='coerce')
        hits_df['distance']  = pd.to_numeric(hits_df['distance'],  errors='coerce')
        hits_df = hits_df.dropna(subset=['direction', 'distance']).reset_index(drop=True)

        if len(hits_df):
            ang = np.radians(hits_df['direction'].to_numpy())
            sx  = hits_df['distance'].to_numpy() * np.sin(ang)
            sy  = hits_df['distance'].to_numpy() * np.cos(ang)

            # Small jitter so overlapping hits don't stack invisibly
            rng    = np.random.default_rng(seed=42)
            jitter = 6  # feet
            sx = sx + rng.uniform(-jitter, jitter, size=len(sx))
            sy = sy + rng.uniform(-jitter, jitter, size=len(sy))

            # Field lines
            for angle in [-45, 45]:
                rad = np.radians(angle)
                ax_spray.plot([0, 330*np.sin(rad)], [0, 330*np.cos(rad)],
                              'w-', lw=1.2, alpha=0.7)
            theta = np.linspace(-np.radians(45), np.radians(45), 200)
            wall  = 330 + 70 * np.cos(2*theta)
            ax_spray.plot(wall*np.sin(theta), wall*np.cos(theta), 'w-', lw=1.8)
            theta_if = np.linspace(-np.radians(45), np.radians(45), 200)
            ax_spray.plot(95*np.sin(theta_if), 95*np.cos(theta_if),
                          'w--', lw=0.8, alpha=0.5)

            for hit_type, style in hit_type_styles.items():
                # .to_numpy() ensures positional boolean indexing into sx/sy
                mask = (hits_df['playresult'] == hit_type).to_numpy()
                if mask.sum():
                    ax_spray.scatter(sx[mask], sy[mask],
                                     color=style['color'], marker=style['marker'],
                                     s=70 if hit_type == 'HomeRun' else 45,
                                     edgecolors='white', linewidths=0.4,
                                     label=f"{hit_type} ({mask.sum()})", zorder=3)

        ax_spray.legend(loc='upper right', fontsize=6, framealpha=0.85,
                        title='Hit Type', title_fontsize=7)
        ax_spray.set_xlim(-400, 400); ax_spray.set_ylim(-50, 450)
        ax_spray.set_aspect('equal')
        ax_spray.set_title('Spray Chart', fontsize=10, fontweight='bold', pad=6)
        ax_spray.axis('off')

        # ── SWING HEAT MAP ───────────────────────────────────────────────────
        ax_heat = fig.add_subplot(gs[2, 1])

        sw = df[df['pitchcall'].isin(['InPlay','StrikeSwinging',
                                       'FoulBallNotFieldable','FoulBall'])].copy()
        sw = sw.dropna(subset=['platelocside','platelocheight'])

        if len(sw):
            hmap, xe, ye = np.histogram2d(
                sw['platelocside'], sw['platelocheight'],
                bins=30, range=[[-2.5, 2.5],[0, 5]])
            hmap = gaussian_filter(hmap, sigma=1.5)
            extent = [xe[0], xe[-1], ye[0], ye[-1]]
            im = ax_heat.imshow(hmap.T, extent=extent, origin='lower',
                                cmap='RdYlGn_r', aspect='auto', alpha=0.88)
            plt.colorbar(im, ax=ax_heat, label='Swing Frequency', shrink=0.8)

        draw_strike_zone(ax_heat)
        ax_heat.set_xlim(-2.5, 2.5); ax_heat.set_ylim(0, 5)
        ax_heat.invert_xaxis()
        ax_heat.set_ylabel("Height (ft)", fontsize=9)
        ax_heat.set_title("Swing Heat Map", fontsize=10, fontweight='bold', pad=6)

        # ── PITCH TYPE BAR CHART ─────────────────────────────────────────────
        # Use _pitch_type so counts are guaranteed to match the table below
        pt_counts = (df[df['_pitch_type'] != 'Unknown']['_pitch_type']
                     .value_counts()
                     .sort_values(ascending=False))
        ax_pitch = fig.add_subplot(gs[3, :])
        colors = [pitch_type_styles.get(p, '#95a5a6') for p in pt_counts.index]
        bars = ax_pitch.bar(pt_counts.index, pt_counts.values, color=colors,
                            edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars, pt_counts.values):
            ax_pitch.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.3, str(val),
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_pitch.set_title('Pitches Seen by Type', fontsize=10, fontweight='bold')
        ax_pitch.set_xlabel('Pitch Type', fontsize=8)
        ax_pitch.set_ylabel('Count', fontsize=8)
        ax_pitch.set_xticklabels(pt_counts.index, rotation=30, ha='right', fontsize=7)
        ax_pitch.set_facecolor('#fafafa')
        ax_pitch.spines['top'].set_visible(False)
        ax_pitch.spines['right'].set_visible(False)

        # ── STATS BY PITCH TYPE TABLE ────────────────────────────────────────
        ax_pt_table = fig.add_subplot(gs[4, :])
        ax_pt_table.axis('off')

        # Group by the same _pitch_type column so counts always match the bar chart
        pitch_groups = (df[df['_pitch_type'] != 'Unknown']
                        .groupby('_pitch_type', sort=False))
        rows = []
        for pitch, group in pitch_groups:
            pt_pa    = (group['korbb'].isin(['Strikeout','Walk']) |
                        group['playresult'].isin(['Single','Double','Triple','HomeRun',
                                                'Out','Error','FieldersChoice','Sacrifice']) |
                        group['pitchcall'].eq('HitByPitch')).sum()
            pt_ab    = (group['korbb'].eq('Strikeout') |
                        group['playresult'].isin(['Single','Double','Triple','HomeRun',
                                                'Out','Error','FieldersChoice'])).sum()
            pt_h     = group['playresult'].isin(['Single','Double','Triple','HomeRun']).sum()
            pt_hr    = group['playresult'].eq('HomeRun').sum()
            pt_bb    = group['korbb'].eq('Walk').sum()
            pt_k     = group['korbb'].eq('Strikeout').sum()
            pt_swing = group['pitchcall'].isin(['StrikeSwinging','FoulBallNotFieldable',
                                                'FoulBall','InPlay']).sum()
            pt_miss  = group['pitchcall'].eq('StrikeSwinging').sum()

            pt_avg    = pt_h / pt_ab  if pt_ab    else 0.0
            pt_bb_pct = pt_bb / pt_pa * 100 if pt_pa    else 0.0
            pt_k_pct  = pt_k  / pt_pa * 100 if pt_pa    else 0.0
            pt_whiff  = pt_miss / pt_swing * 100 if pt_swing else 0.0

            pt_ev = pd.to_numeric(group[group['pitchcall']=='InPlay']['exitspeed'],
                                  errors='coerce').dropna()
            pt_bs = pd.to_numeric(group['batspeed'], errors='coerce').dropna()

            rows.append({
                'Pitch Type': pitch,
                'Pitches':    len(group),   # total pitches — matches bar chart
                'PA':         pt_pa,
                'AB':         pt_ab,
                'H':          pt_h,
                'HR':         pt_hr,
                'AVG':        f'{pt_avg:.3f}',
                'BB%':        f'{pt_bb_pct:.1f}%',
                'K%':         f'{pt_k_pct:.1f}%',
                'Whiff%':     f'{pt_whiff:.1f}%',
                'Avg EV':     f'{pt_ev.mean():.1f}' if len(pt_ev) else 'N/A',
                'Avg BatSpd': f'{pt_bs.mean():.1f}' if len(pt_bs) else 'N/A',
            })

        pt_df = pd.DataFrame(rows).sort_values('Pitches', ascending=False).reset_index(drop=True)
        col_labels_tbl = list(pt_df.columns)

        tbl = ax_pt_table.table(
            cellText=pt_df.values.tolist(),
            colLabels=col_labels_tbl,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

        for j in range(len(col_labels_tbl)):
            tbl[0, j].set_facecolor(SCHOOL_COLOR)
            tbl[0, j].set_text_props(color='black', fontweight='bold')
            tbl[0, j].set_edgecolor('white')

        for i in range(1, len(pt_df) + 1):
            for j in range(len(col_labels_tbl)):
                tbl[i, j].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                tbl[i, j].set_edgecolor('#dddddd')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN: loop over all batters
# ══════════════════════════════════════════════════════════════════════════════
def safe_filename(name):
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_all = pd.read_csv(CSV_FILE, low_memory=False)
    df_all.columns = df_all.columns.str.lower().str.strip()

    batters = df_all['batter'].dropna().unique()
    print(f"Found {len(batters)} unique batters. Generating reports...\n")

    for i, batter_name in enumerate(sorted(batters), 1):
        batter_df = df_all[df_all['batter'].str.lower() == batter_name.lower()].copy()
        fname = os.path.join(OUTPUT_DIR, f"{safe_filename(batter_name)}.pdf")
        print(f"[{i}/{len(batters)}] {batter_name} — {len(batter_df)} pitches → {fname}")
        try:
            generate_report(batter_df, batter_name, fname)
        except Exception as e:
            print(f"  ⚠️  Skipped {batter_name}: {e}")

    print(f"\n✅  All reports saved to '{OUTPUT_DIR}/' folder.")


if __name__ == '__main__':
    main()