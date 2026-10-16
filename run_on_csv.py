"""Run DeepSPT temporal segmentation on a tracks CSV (columns: particle, frame, x, y).

    python run_on_csv.py test_data/tracks_1_h_2_ugml.csv --dt 0.02 --xy-to-um 0.001

Writes <out>/<csvname>_frames.csv (per-frame class + probabilities) and
<out>/<csvname>_tracks.csv (per-track class fractions). Run from the repo root.
"""
import argparse, os
import numpy as np, pandas as pd
from deepspt_src import *

CLASSES = ['Normal', 'Directed', 'Confined', 'Subdiffusive']

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--dt', type=float, default=0.02, help='frame interval in seconds (unused by the model, kept for the record)')
ap.add_argument('--xy-to-um', type=float, default=0.001, help='scale x,y to micrometres; nm input -> 0.001, pixels -> pixel size in um')
ap.add_argument('--min-len', type=int, default=5, help='drop tracks shorter than this (model trained on 5-600 frames)')
ap.add_argument('--out', default='deepspt_results')
ap.add_argument('--cpu', action='store_true')
a = ap.parse_args()

device = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
df = pd.read_csv(a.csv).sort_values(['particle', 'frame'])
groups = [(pid, g) for pid, g in df.groupby('particle') if len(g) >= a.min_len]
tracks = [g[['x', 'y']].values.astype(float) * a.xy_to_um for _, g in groups]
print(f'{len(tracks)} tracks of {df.particle.nunique()} kept (>= {a.min_len} frames), device={device}')

# same preprocessing as usage_example1.py: start at origin, add step-length + dot-product features
X = [x - x[0] for x in tracks]
X_to_eval = add_features(X, ['XYZ', 'SL', 'DP'])
y_to_eval = [np.ones(len(x)) * 0.5 for x in X_to_eval]

modeldir = 'mlruns/3'  # 2D ensemble; '36' is the 3D one
assert os.path.exists(modeldir), 'mlruns/3 missing - run from the repo root with the models in place'
best_models_sorted = find_models_for_from_path(modeldir)

os.makedirs(a.out, exist_ok=True)
stem = os.path.join(a.out, os.path.splitext(os.path.basename(a.csv))[0])
ensemble_score, ensemble_pred = run_temporalsegmentation(
    best_models_sorted, X_to_eval, y_to_eval, use_mlflow=False, dir_name='', device=device,
    dim=2, min_max_len=601, X_padtoken=0, y_padtoken=10, batch_size=32, rerun_segmentaion=True,
    savename_score=stem + '_ensemble_score.pkl', savename_pred=stem + '_ensemble_pred.pkl',
    use_temperature=True)

frames = []
for (pid, g), score, pred in zip(groups, ensemble_score, ensemble_pred):
    out = g[['particle', 'frame', 'x', 'y']].copy()
    out['pred'] = pred
    out['class'] = [CLASSES[p] for p in pred]
    for k, c in enumerate(CLASSES):
        out['p_' + c] = score[k]
    frames.append(out)
frames = pd.concat(frames)
frames.to_csv(stem + '_frames.csv', index=False)

per_track = frames.groupby('particle')['class'].value_counts(normalize=True).unstack(fill_value=0)
per_track['n_frames'] = frames.groupby('particle').size()
per_track['dt_s'] = a.dt
per_track.to_csv(stem + '_tracks.csv')
print(frames['class'].value_counts(normalize=True).round(3).to_string())
print('wrote', stem + '_frames.csv', 'and', stem + '_tracks.csv')
