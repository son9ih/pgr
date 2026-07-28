import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

DATA_DIR = 'ablation_data'
SAVE_DIR = 'ablation_figures'
os.makedirs(SAVE_DIR, exist_ok=True)

ENV_DISPLAY = {
    'HalfCheetah-v2': 'HalfCheetah-v2',
    'Hopper-v2': 'Hopper-v2',
    'reacher-hard-v0': 'Reacher-Hard-v0',
}

ALPHA_ORDER = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALPHA_COLORS = {
    0.1:  '#e41a1c',
    0.5:  '#ff7f00',
    1.0:  '#4daf4a',
    2.0:  '#377eb8',
    5.0:  '#984ea3',
    10.0: '#a65628',
}


def load_all_data():
    envs = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        env = f.replace('gda-for-orl_Abl_OnpolicyReward_', '').replace('.parquet', '')
        df = pd.read_parquet(os.path.join(DATA_DIR, f))
        df['alpha_rtb'] = df['alpha_rtb'].astype(float)
        envs[env] = df
    return envs


def plot_learning_curves(envs):
    """One subplot per environment; mean +/- std over seeds for each alpha_rtb."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (env, df) in zip(axes, envs.items()):
        for alpha in ALPHA_ORDER:
            sub = df[df['alpha_rtb'] == alpha]
            grouped = sub.groupby('Training_Epoch')['OnPolicy_Reward']
            mean = grouped.mean()
            std = grouped.std()
            epochs = mean.index.values

            ax.plot(epochs, mean.values, label=rf'$\alpha_{{rtb}}={alpha}$',
                    color=ALPHA_COLORS[alpha], linewidth=1.8)
            ax.fill_between(epochs, (mean - std).values, (mean + std).values,
                            color=ALPHA_COLORS[alpha], alpha=0.15)

        ax.set_title(ENV_DISPLAY.get(env, env), fontweight='bold')
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('On-Policy Reward')

    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    frameon=True, fancybox=True, shadow=False)
    fig.suptitle('Ablation: On-Policy Reward vs Training Epoch', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'learning_curves.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def plot_final_reward_bar(envs):
    """Grouped bar chart of mean final-epoch reward per alpha per environment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (env, df) in zip(axes, envs.items()):
        max_epoch = df['Training_Epoch'].max()
        final = df[df['Training_Epoch'] == max_epoch]

        means, stds = [], []
        for alpha in ALPHA_ORDER:
            vals = final[final['alpha_rtb'] == alpha]['OnPolicy_Reward']
            means.append(vals.mean())
            stds.append(vals.std())

        x = np.arange(len(ALPHA_ORDER))
        colors = [ALPHA_COLORS[a] for a in ALPHA_ORDER]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black',
                       linewidth=0.6, alpha=0.85, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ALPHA_ORDER])
        ax.set_xlabel(r'$\alpha_{rtb}$')
        ax.set_ylabel('On-Policy Reward (final epoch)')
        ax.set_title(ENV_DISPLAY.get(env, env), fontweight='bold')

    fig.suptitle('Final Epoch On-Policy Reward by $\\alpha_{rtb}$',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'final_reward_bar.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def plot_individual_runs(envs):
    """Per-environment figure showing every individual run, colored by alpha_rtb."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (env, df) in zip(axes, envs.items()):
        for alpha in ALPHA_ORDER:
            sub = df[df['alpha_rtb'] == alpha]
            for run_id, run_df in sub.groupby('run_id'):
                run_df = run_df.sort_values('Training_Epoch')
                ax.plot(run_df['Training_Epoch'].values, run_df['OnPolicy_Reward'].values,
                        color=ALPHA_COLORS[alpha], alpha=0.5, linewidth=0.9)

            grouped = sub.groupby('Training_Epoch')['OnPolicy_Reward'].mean()
            ax.plot(grouped.index, grouped.values, color=ALPHA_COLORS[alpha],
                    linewidth=2.2, label=rf'$\alpha_{{rtb}}={alpha}$')

        ax.set_title(ENV_DISPLAY.get(env, env), fontweight='bold')
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('On-Policy Reward')

    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    frameon=True, fancybox=True, shadow=False,
                    title='Mean (bold)')
    fig.suptitle('Individual Runs + Mean by $\\alpha_{rtb}$',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'individual_runs.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def plot_heatmap(envs):
    """Heatmap of mean reward over (alpha_rtb, epoch_bin) for each environment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (env, df) in zip(axes, envs.items()):
        pivot = df.pivot_table(index='alpha_rtb', columns='Training_Epoch',
                               values='OnPolicy_Reward', aggfunc='mean')
        pivot = pivot.loc[ALPHA_ORDER]

        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean Reward', fontsize=10)

        ax.set_yticks(range(len(ALPHA_ORDER)))
        ax.set_yticklabels([str(a) for a in ALPHA_ORDER])
        ax.set_ylabel(r'$\alpha_{rtb}$')

        epoch_vals = pivot.columns.values
        tick_positions = np.linspace(0, len(epoch_vals) - 1, 6, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(epoch_vals[tick_positions])
        ax.set_xlabel('Training Epoch')
        ax.set_title(ENV_DISPLAY.get(env, env), fontweight='bold')

    fig.suptitle('Mean On-Policy Reward Heatmap',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def plot_summary_table(envs):
    """Prints and saves a summary table to CSV."""
    rows = []
    for env, df in envs.items():
        max_epoch = df['Training_Epoch'].max()
        for alpha in ALPHA_ORDER:
            sub = df[df['alpha_rtb'] == alpha]
            final = sub[sub['Training_Epoch'] == max_epoch]['OnPolicy_Reward']
            overall = sub['OnPolicy_Reward']
            rows.append({
                'Environment': ENV_DISPLAY.get(env, env),
                'alpha_rtb': alpha,
                'Mean_Reward_All': overall.mean(),
                'Std_Reward_All': overall.std(),
                'Mean_Final_Reward': final.mean(),
                'Std_Final_Reward': final.std(),
                'Min_Final_Reward': final.min(),
                'Max_Final_Reward': final.max(),
                'Num_Seeds': sub['run_id'].nunique(),
            })
    summary = pd.DataFrame(rows)
    path = os.path.join(SAVE_DIR, 'summary_table.csv')
    summary.to_csv(path, index=False, float_format='%.6f')
    print(f'Saved {path}')
    print('\n' + summary.to_string(index=False, float_format='{:.6f}'.format))


if __name__ == '__main__':
    envs = load_all_data()
    print(f'Loaded {len(envs)} environments: {list(envs.keys())}')
    for env, df in envs.items():
        print(f'  {env}: {len(df)} rows, {df.run_id.nunique()} runs, '
              f'alphas={sorted(df.alpha_rtb.unique())}')
    print()

    plot_learning_curves(envs)
    plot_final_reward_bar(envs)
    plot_individual_runs(envs)
    plot_heatmap(envs)
    plot_summary_table(envs)

    print(f'\nAll figures saved to {SAVE_DIR}/')
