"""Visualise run_on_csv.py output.

    python plot_results.py deepspt_results/tracks_1_h_2_ugml_frames.csv [--track 0] [--min-len 20]

Writes <stem>_overview.png: tracks coloured by diffusion class, class fractions,
and one track's class probabilities over time.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['Normal', 'Directed', 'Confined', 'Subdiffusive']
COLORS = dict(zip(CLASSES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))  # fixed order, CVD-validated

ap = argparse.ArgumentParser()
ap.add_argument('frames_csv')
ap.add_argument('--track', type=int, help='particle id for the time panel (default: longest track)')
ap.add_argument('--min-len', type=int, default=20, help='only draw tracks with at least this many frames in the map')
a = ap.parse_args()

df = pd.read_csv(a.frames_csv)
lens = df.groupby('particle').size()
pid = a.track if a.track is not None else lens.idxmax()
tr = df[df.particle == pid]
stem = a.frames_csv.replace('_frames.csv', '')

fig, ax = plt.subplots(2, 2, figsize=(13, 10), facecolor='#fcfcfb')
fig.suptitle(os.path.basename(stem), color='#333')

# 1. all tracks, each segment coloured by its class (um)
m = ax[0, 0]
for p, g in df[df.particle.isin(lens[lens >= a.min_len].index)].groupby('particle'):
    x, y = g.x.values / 1000, g.y.values / 1000
    for c in CLASSES:
        idx = np.where(g['class'].values == c)[0]
        m.plot(x[idx], y[idx], '.', ms=2, color=COLORS[c], alpha=.7)
m.set(xlabel='x (um)', ylabel='y (um)', title=f'Tracks >= {a.min_len} frames, coloured by class', aspect='equal')
m.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=COLORS[c], label=c) for c in CLASSES], frameon=False, fontsize=8)

# 2. class fractions, all frames
b = ax[0, 1]
frac = df['class'].value_counts(normalize=True).reindex(CLASSES).fillna(0)
bars = b.bar(CLASSES, frac.values, color=[COLORS[c] for c in CLASSES], width=.6)
b.bar_label(bars, labels=[f'{v:.1%}' for v in frac.values], color='#333', padding=3)
b.set(ylabel='fraction of frames', title='Class fractions', ylim=(0, frac.max() * 1.2))

# 3. one track in xy, coloured by class
t = ax[1, 0]
x, y = (tr.x.values - tr.x.values[0]) / 1000, (tr.y.values - tr.y.values[0]) / 1000
t.plot(x, y, '-', color='#ccc', lw=1, zorder=0)
for c in CLASSES:
    idx = np.where(tr['class'].values == c)[0]
    t.plot(x[idx], y[idx], '.', ms=5, color=COLORS[c])
t.set(xlabel='x (um)', ylabel='y (um)', title=f'Track {pid}: {len(tr)} frames', aspect='equal')

# 4. same track: ensemble probability per class over time
s = ax[1, 1]
for c in CLASSES:
    s.plot(tr.frame.values, tr['p_' + c].values, lw=2, color=COLORS[c], label=c)
s.set(xlabel='frame', ylabel='ensemble probability', ylim=(0, 1.02), title=f'Track {pid}: class probability over time')
s.legend(frameon=False, fontsize=8, ncol=4, loc='upper center')

for A in ax.flat:
    A.set_facecolor('#fcfcfb'); A.grid(alpha=.25); [A.spines[k].set_visible(False) for k in ('top', 'right')]
fig.tight_layout()
fig.savefig(stem + '_overview.png', dpi=130)
print('wrote', stem + '_overview.png')
