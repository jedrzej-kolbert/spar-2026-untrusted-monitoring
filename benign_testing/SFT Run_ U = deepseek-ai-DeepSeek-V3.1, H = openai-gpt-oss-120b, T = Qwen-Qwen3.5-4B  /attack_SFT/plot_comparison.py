"""Overlay completed exports without model calls or re-optimization.

Run: python plot_comparison.py [--root PATH] [--output PATH]
Requires the existing project's matplotlib, numpy and scikit-learn dependencies.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import gzip
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root, output = args.root, args.output or args.root
output.mkdir(parents=True, exist_ok=True)
conditions = [('Base', 'attack_SFT_full_test_batch64', '#0072B2'),
              ('Minified', 'attack_SFT_minified_full_test_batch64', '#D55E00'),
              ('Paraphrased', 'attack_SFT_paraphrased_full_test_batch64_dynamics', '#009E73')]
loaded = []
provenance = []
for label, folder, color in conditions:
    path = root / folder / 'analysis_data.json.gz'
    with gzip.open(path, 'rt') as file:
        payload = json.load(file)
    assert payload['converged'], f'{label}: optimization not converged'
    assert payload['sample_counts'] == {'attacks': 114, 'honeypots': 246, 'benign': 751}
    if loaded:
        assert payload['game_parameters'] == loaded[0][2]['game_parameters'], 'Game parameters differ'
    loaded.append((label, color, payload))
    provenance.append({'condition': label, 'source': f'{folder}/analysis_data.json.gz',
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for extension in ('svg', 'png'):
        fig.savefig(output / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    ET.parse(output / f'{name}.svg')
    plt.close(fig)

roc_rows = []
for key, title, filename, xlabel, ylabel in [
    ('untrusted_collusion', 'Untrusted collusion ROC', 'untrusted_monitor_roc_curve',
     'False positive rate (honeypots)', 'True positive rate (attacks)'),
    ('trusted_suspiciousness', 'Trusted suspiciousness ROC', 'trusted_monitor_roc_curve',
     'False positive rate (honest solutions)', 'True positive rate (attacks)'),
    ('untrusted_suspiciousness', 'Untrusted suspiciousness ROC', 'untrusted_suspiciousness_roc_curve',
     'False positive rate (honest solutions)', 'True positive rate (attacks)'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, color, payload in loaded:
        curve = payload['roc_curves'][key]
        x, y = np.asarray(curve['fpr']), np.asarray(curve['tpr'])
        assert len(x) == len(y) == len(curve['thresholds'])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert ((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1)).all()
        assert (np.diff(x) <= 0).all() and (np.diff(y) <= 0).all()
        value = auc(x, y)
        for ax in axes:
            ax.plot(x, y, color=color, linewidth=2, label=f'{label} (AUC {value:.4f})')
        roc_rows.extend({'condition': label, 'monitor': key, 'threshold': threshold,
                         'fpr': fpr, 'tpr': tpr, 'auc': value}
                        for threshold, fpr, tpr in zip(curve['thresholds'], x, y))
        print(f'{key}: {label} AUC {value:.6f}')
    for ax, xmax, panel in zip(axes, (1, .05), ('Full range', 'Low false positive rate')):
        ax.plot([0, xmax], [0, xmax], '--', color='#999999', linewidth=1)
        ax.set(xlim=(0, xmax), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel, title=panel)
        ax.grid(alpha=.2)
    assert all(len(ax.lines) == 4 for ax in axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'{title} — full APPS test, attack SFT, batch 64')
    save(fig, filename)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
safety_rows = []
for label, color, payload in loaded:
    points = payload['sweep']
    x = [p['safety'] for p in points]
    y = [p['usefulness'] for p in points]
    assert np.isfinite(x + y).all() and all(0 <= v <= 1 for v in x + y)
    assert len(points) == 10
    for ax in axes:
        ax.plot(x, y, 'o-', color=color, linewidth=2, markersize=4, label=label)
    safety_rows.extend({'condition': label, 'safety': p['safety'], 'usefulness': p['usefulness']}
                       for p in points)
for ax in axes:
    ax.set(xlabel='Safety (blue-team win rate)', ylabel='Usefulness (1 − replacement FNR)')
    ax.grid(alpha=.2)
axes[0].set(xlim=(0, 1), ylim=(0, 1.02), title='Full range')
axes[1].set(xlim=(.1, .45), ylim=(.8, 1.005), title='High usefulness (0.8–1.0)')
assert all(len(ax.lines) == 3 for ax in axes)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
fig.suptitle('Safety vs usefulness — full APPS test, attack SFT, batch 64')
save(fig, 'safety_usefulness_curves')
for filename, rows in [('comparison_roc_points.csv', roc_rows), ('comparison_safety_usefulness_points.csv', safety_rows)]:
    with (output / filename).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(output / 'comparison_sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('Verified three conditions per chart, source compatibility, curve coordinates and SVG validity.')
