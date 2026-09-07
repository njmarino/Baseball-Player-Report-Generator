import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

# ── SETTINGS ──────────────────────────────────────────────────────────────────
CSV_FILE    = 'NDSU_Hitting.csv'         # path to your TrackMan CSV
BATTER_NAME = 'Woita, Jase'          # batter to report on
OUTPUT_FILE = 'report.pdf'            # output file name

hit_type_styles = {
    'Single':       {'color': '#2ecc71', 'marker': 'o', 'zorder': 3},
    'Double':       {'color': '#3498db', 'marker': 'o', 'zorder': 3},
    'Triple':       {'color': '#9b59b6', 'marker': 'o', 'zorder': 3},
    'HomeRun':      {'color': '#e74c3c', 'marker': '*', 'zorder': 4},
    'Out':          {'color': '#95a5a6', 'marker': 'x', 'zorder': 2},
    'FieldersChoice':{'color': '#f39c12', 'marker': 'o', 'zorder': 3},
    'Error':        {'color': '#e67e22', 'marker': 'D', 'zorder': 3},
}

pitch_type_styles = {
    'Fastball':             {'color': '#e74c3c', 'marker': 'o', 'zorder': 3},
    'Slider':             {'color': '#C9A227', 'marker': 'o', 'zorder': 3},
    'Sinker':             {'color': '#e67e22', 'marker': 'o', 'zorder': 3},
    'Cutter':             {'color': '#9b59b6', 'marker': 'o', 'zorder': 3},
    'Curveball':          {'color': '#3498db', 'marker': 'o', 'zorder': 3},
    'Changeup':           {'color': '#2ecc71', 'marker': 'o', 'zorder': 3},
    'Splitter':           {'color': '#e67e22', 'marker': 'o', 'zorder': 3},
}

# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(CSV_FILE, low_memory=False)
df.columns = df.columns.str.lower().str.strip()

# Filter down to just this batter
df = df[df['batter'].str.lower() == BATTER_NAME.lower()].copy()
print(f"Loaded {len(df)} pitches for {BATTER_NAME}")

# ══════════════════════════════════════════════════════════════════════════════
#  2. CALCULATE STATS
# ══════════════════════════════════════════════════════════════════════════════

# --- Plate appearances, at-bats, hits ---
pa  = (df['korbb'].isin(['Strikeout', 'Walk']) |
       df['playresult'].isin(['Single','Double','Triple','HomeRun','Out','Error','FieldersChoice','Sacrifice']) |
       df['pitchcall'].eq('HitByPitch')).sum()

ab  = (df['korbb'].eq('Strikeout') |
       df['playresult'].isin(['Single','Double','Triple','HomeRun','Out','Error','FieldersChoice'])).sum()

h   = df['playresult'].isin(['Single','Double','Triple','HomeRun']).sum()
dbl = df['playresult'].eq('Double').sum()
tpl = df['playresult'].eq('Triple').sum()
hr  = df['playresult'].eq('HomeRun').sum()
bb  = df['korbb'].eq('Walk').sum()
k   = df['korbb'].eq('Strikeout').sum()
hbp = df['pitchcall'].eq('HitByPitch').sum()

# --- Rate stats ---
avg = h / ab                         if ab else 0.0
obp = (h + bb + hbp) / pa            if pa else 0.0
slg = (h + dbl + 2*tpl + 3*hr) / ab  if ab else 0.0
ops = obp + slg

# --- Plate discipline ---
swings = df['pitchcall'].isin({'StrikeSwinging','FoulBallNotFieldable','FoulBall','InPlay'}).sum()
misses = df['pitchcall'].eq('StrikeSwinging').sum()
whiff  = misses / swings if swings else 0.0

# --- Exit velocity ---
inplay = df[df['pitchcall'] == 'InPlay']
ev     = pd.to_numeric(inplay['exitspeed'], errors='coerce').dropna()
avg_ev = ev.mean() if len(ev) else None

# ══════════════════════════════════════════════════════════════════════════════
#  3. Spray Chart
# ══════════════════════════════════════════════════════════════════════════════
hits = df[df['pitchcall'] == 'InPlay'].copy()
#Convert polar to cartesian coordinates
angles_rad = np.radians(hits['direction'])
x = hits['distance'] * np.sin(angles_rad)
y = hits['distance'] * np.cos(angles_rad)

fig, ax = plt.subplots(figsize=(10, 10))

# Draw the field
for angle in [-45, 45]:
    rad = np.radians(angle)
    ax.plot([0, 330 * np.sin(rad)], [0, 330 * np.cos(rad)], 'k-', lw=1.5)
#outfield arc
theta = np.linspace(-np.radians(45), np.radians(45), 200)
wall_distances = 330 + 70 * np.cos(2 * theta)  # approximates typical wall shape
wx = wall_distances * np.sin(theta)
wy = wall_distances * np.cos(theta)
ax.plot(wx, wy, 'k-', lw=2)
# Infield arc (95 ft)
theta_if = np.linspace(np.radians(-45), np.radians(45), 200)
ax.plot(95 * np.sin(theta_if), 95 * np.cos(theta_if), 'k--', lw=1, alpha=0.4)
# --- Plot hits, colored by exit velocity or hit type ---

for hit_type, style in hit_type_styles.items():
    mask = df['playresult'] == hit_type
    if mask.sum() > 0:
        ax.scatter(x[mask], y[mask], color=style['color'], marker=style['marker'], s=80, edgecolor='white', label=hit_type, linewidths=0.4)

plt.legend(loc='upper right', fontsize=10, framealpha=0.9, title='Hit Type', title_fontsize=11)

ax.set_xlim(-400, 400)
ax.set_ylim(-50, 450)
ax.set_facecolor('#c8e6a0')  # grass green
ax.set_title('Spray Chart', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  3. Swing Heat Map
# ══════════════════════════════════════════════════════════════════════════════
swings = df[df['pitchcall'].isin(['InPlay', 'StrikeSwinging', 'FoulBallNotFieldable'])].copy()
# Drop rows missing plate location
swings = swings.dropna(subset=['platelocside', 'platelocheight'])
x = swings['platelocside']      # horizontal location (ft), negative = inside to RHH
y = swings['platelocheight']    # vertical location (ft)

# --- Setup figure ---
fig, ax = plt.subplots(figsize=(7, 9))

# --- Heatmap via 2D histogram + gaussian blur ---
heatmap, xedges, yedges = np.histogram2d(
    x, y,
    bins=30,
    range=[[-2.5, 2.5], [0, 5]]
)
heatmap = gaussian_filter(heatmap, sigma=1.5)  # smooth it out

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax.imshow(
    heatmap.T,
    extent=extent,
    origin='lower',
    cmap='RdYlGn_r',   # red = high density, green = low
    aspect='auto',
    alpha=0.85
)
plt.colorbar(im, ax=ax, label='Swing Frequency')

# --- Strike zone box ---
# Standard zone: roughly 17 inches wide, sz_bot to sz_top
zone_left  = -0.708   # 17in / 2 in feet
zone_right =  0.708
sz_bot = 1.5
sz_top = 3.5

zone = plt.Rectangle(
    (zone_left, sz_bot),
    zone_right - zone_left,
    sz_top - sz_bot,
    linewidth=2, edgecolor='black', facecolor='none', zorder=5
)
ax.add_patch(zone)

# Inner zone thirds (3x3 grid)
for x_line in [-0.236, 0.236]:
    ax.plot([x_line, x_line], [sz_bot, sz_top], 'k-', lw=0.8, alpha=0.5)
for y_line in [2.167, 2.833]:
    ax.plot([zone_left, zone_right], [y_line, y_line], 'k-', lw=0.8, alpha=0.5)

# Home plate outline
plate_x = [-0.708, 0.708, 0.708, 0, -0.708, -0.708]
plate_y = [0.5, 0.5, 0.35, 0.1, 0.35, 0.5]
ax.plot(plate_x, plate_y, 'k-', lw=1.5)

# --- Labels ---
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 5)
ax.set_xlabel("Horizontal Location (ft)", fontsize=11)
ax.set_ylabel("Vertical Location (ft)", fontsize=11)
ax.set_title("Swing Heat Map", fontsize=14, fontweight='bold')

# Flip x-axis for catcher's perspective
ax.invert_xaxis()

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  3. Pitches Seen by Type
# ══════════════════════════════════════════════════════════════════════════════
pitch_types = df['taggedpitchtype'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(pitch_types.index, pitch_types.values, color=[pitch_type_styles.get(pt, {}).get('color', '#95a5a6') for pt in pitch_types.index], edgecolor='black')
ax.set_title("Pitches Seen by Type", fontsize=14, fontweight='bold')
ax.set_xlabel("Pitch Type", fontsize=11)   
ax.set_ylabel("Count", fontsize=11)
ax.set_xticklabels(pitch_types.index, rotation=45, ha='right')
plt.tight_layout()
plt.show()
