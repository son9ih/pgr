"""Plot the synthetic-buffer capacity ablation: three learning curves, mean +- std
over seeds, from the CSV written by fetch_curves.py.

Usage:  python exp/tools/plot_buffer_ablation.py
        python exp/tools/plot_buffer_ablation.py --every 7 --smooth 0

Styled to match the paper's existing figures (fine_tuning.pdf): seaborn darkgrid
surface with white gridlines and no spines, serif type, markers on a subsampled
line, and a frameless horizontal legend under the axes.

Color: buffer capacity is ORDINAL (50k < 200k < 1M), so the three series take a
one-hue ramp -- light = small capacity -- rather than categorical hues, and the
ordering is visible in the color. The ramp was re-stepped for the darkgrid
surface: on #EAEAF2 the previous light end (#86b6ef) fell to 1.76:1 and failed
the ordinal floor. #3987e5 -> #1c5cab -> #0d366b passes every ordinal check
(monotone lightness, min adjacent dL 0.141, light end 3.04:1) and all three
steps clear 3:1, so the legend alone carries identity -- no relief labels needed.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, '..', 'data', 'buffer_ablation_reacher.csv')
DEFAULT_OUT = os.path.join(HERE, '..', 'figures', 'buffer_ablation_reacher')

# (csv label, color, legend label) -- light -> dark = small -> large capacity
ARMS = [('50k', '#3987e5', 'Buffer 50K'),
        ('200k', '#1c5cab', 'Buffer 200K'),
        ('1M', '#0d366b', 'Buffer 1M')]


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
    wide = wide[wide.notna().sum(axis=1) >= max(1, wide.shape[1] // 2)]
    sd = wide.std(axis=1, ddof=1).to_numpy() if wide.shape[1] > 1 else np.zeros(len(wide))
    return wide.index.to_numpy(), wide.mean(axis=1).to_numpy(), sd, wide.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--out', default=DEFAULT_OUT, help='path without extension')
    ap.add_argument('--smooth', type=int, default=0, help='rolling-mean window; 0 = raw')
    ap.add_argument('--every', type=int, default=7,
                    help='draw every Nth epoch (matches the reference figure density)')
    ap.add_argument('--title', default='Reacher-Hard')
    a = ap.parse_args()

    df = pd.read_csv(os.path.abspath(a.csv))
    sns.set_theme(style='darkgrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.titlesize': 20,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 15,
    })
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    table = []
    for label, color, pretty in ARMS:
        st = arm_stats(df, label, a.smooth)
        if st is None:
            print(f'  (no data for {label}, skipping)')
            continue
        x, mu, sd, n = st
        # Subsample for drawing only; the table below still reports every epoch.
        # Evenly spaced and landing exactly on the last epoch -- np.arange plus a
        # forced final index puts two points one epoch apart and draws a fake spike.
        k = max(1, a.every)
        sel = np.unique(np.linspace(0, len(x) - 1,
                                    max(2, round((len(x) - 1) / k) + 1)).round().astype(int))
        xs, ms, ss = x[sel], mu[sel], sd[sel]
        ax.fill_between(xs, ms - ss, ms + ss, color=color, alpha=0.20, linewidth=0)
        ax.plot(xs, ms, color=color, linewidth=2.0, marker='o', markersize=5,
                markeredgewidth=0, label=pretty, solid_capstyle='round')
        tail = slice(max(0, len(x) - 20), len(x))
        table.append((pretty, n, mu[-1], sd[-1], float(np.mean(mu[tail]))))

    ax.set_title(a.title)
    ax.set_xlabel('Environment Steps (K)')
    ax.set_ylabel('Average Return')
    ax.set_xlim(0, None)
    # Frameless horizontal legend under the axes, as in the reference figure.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
        print(f'wrote {out}.{ext}')

    # Table view twin -- every plotted value is readable without the colors.
    print(f'\n{"capacity":>12} {"seeds":>6} {"final":>16} {"last-20 mean":>14}')
    for pretty, n, mu, sd, tail in table:
        print(f'{pretty:>12} {n:>6} {mu:>9.1f} ± {sd:<5.1f} {tail:>14.1f}')


if __name__ == '__main__':
    main()
