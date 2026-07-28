"""Plot the synthetic-buffer capacity ablation: three learning curves, mean +- std
over seeds, from the CSV written by fetch_curves.py.

Usage:  python exp/tools/plot_buffer_ablation.py
        python exp/tools/plot_buffer_ablation.py --smooth 5      # rolling mean, off by default

Design notes (dataviz skill):
- Buffer capacity is ORDINAL (50k < 200k < 1M), so the three series take a one-hue
  ramp -- light = small capacity -- instead of categorical hues, and the reader sees
  the ordering in the color. Validated with the skill's ordinal checks: monotone
  lightness, min adjacent dL 0.189 (>= 0.06), light end #86b6ef at 2.06:1 vs the
  #fcfcfb surface (>= 2.0), hue spread 3 deg.
- #86b6ef sits below 3:1, so the relief rule applies: every series is directly
  labeled at its endpoint and the numbers are also printed as a table.
- One y-axis, solid hairline grid, no marker per point, text in ink tokens with the
  colored line carrying identity.
- Light mode only: this renders into a paper, which commits to one surface.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, '..', 'data', 'buffer_ablation_reacher.csv')
DEFAULT_OUT = os.path.join(HERE, '..', 'figures', 'buffer_ablation_reacher')

# Ordinal blue ramp from the skill's palette, light -> dark = small -> large capacity.
ARMS = [('50k', '#86b6ef', '50,000'),
        ('200k', '#2a78d6', '200,000'),
        ('1M', '#104281', '1,000,000')]

SURFACE = '#fcfcfb'
INK_PRIMARY, INK_SECONDARY, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'


def arm_stats(df, label, smooth):
    """Per-epoch mean/std across seeds for one arm."""
    sub = df[df['capacity'] == label]
    if sub.empty:
        return None
    wide = sub.pivot_table(index='epoch', columns='seed', values='return')
    if smooth > 1:
        wide = wide.rolling(smooth, min_periods=1).mean()
    # Keep epochs where at least half the seeds reported, so a short run cannot
    # silently move the mean at the tail.
    keep = wide.notna().sum(axis=1) >= max(1, wide.shape[1] // 2)
    wide = wide[keep]
    return (wide.index.to_numpy(), wide.mean(axis=1).to_numpy(),
            wide.std(axis=1, ddof=1).to_numpy() if wide.shape[1] > 1
            else np.zeros(len(wide)), wide.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--out', default=DEFAULT_OUT, help='path without extension')
    ap.add_argument('--smooth', type=int, default=0, help='rolling-mean window; 0 = raw')
    a = ap.parse_args()

    df = pd.read_csv(os.path.abspath(a.csv))
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 0.8,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    table, ends = [], []
    xmax = 0
    for label, color, pretty in ARMS:
        st = arm_stats(df, label, a.smooth)
        if st is None:
            print(f'  (no data for {label}, skipping)')
            continue
        x, mu, sd, n = st
        xmax = max(xmax, x.max())
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mu, color=color, linewidth=2.0, solid_capstyle='round',
                label=f'{pretty}  (n={n})')
        ends.append((float(mu[-1]), float(x[-1]), pretty))
        tail = slice(max(0, len(x) - 10), len(x))
        table.append((pretty, n, mu[-1], sd[-1], float(np.mean(mu[tail]))))

    ax.set_xlim(0, xmax * 1.10)           # headroom for the endpoint labels
    lo, hi = ax.get_ylim()
    # Direct label at each endpoint, pushed apart so near-equal finals stay legible.
    min_gap = 0.045 * (hi - lo)
    ends.sort()
    placed = []
    for y, x_end, pretty in ends:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, x_end, pretty))
    for y, x_end, pretty in placed:
        ax.annotate(pretty, xy=(x_end, y), xytext=(6, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=9, color=INK_SECONDARY,
                    annotation_clip=False)

    ax.set_xlabel('Epoch (1000 environment steps each)', color=INK_SECONDARY)
    ax.set_ylabel('Evaluation return', color=INK_SECONDARY)
    ax.set_title('Synthetic replay buffer capacity — reacher-hard',
                 color=INK_PRIMARY, fontsize=11.5, pad=12, loc='left')
    ax.grid(axis='y', color=GRID, linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, length=3, width=0.8)
    leg = ax.legend(title='Buffer capacity', frameon=False, loc='lower right',
                    fontsize=9, labelcolor=INK_SECONDARY)
    leg.get_title().set_color(INK_SECONDARY)
    leg.get_title().set_fontsize(9)
    fig.tight_layout()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {out}.{ext}')

    # Table view twin -- every plotted value is readable without the colors.
    print(f'\n{"capacity":>12} {"seeds":>6} {"final":>16} {"last-10 mean":>14}')
    for pretty, n, mu, sd, tail in table:
        print(f'{pretty:>12} {n:>6} {mu:>9.1f} ± {sd:<5.1f} {tail:>14.1f}')


if __name__ == '__main__':
    main()
