import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DATA_DIR = "/home/son9ih/pgr/ablation_data"
SAVE_DIR = "/home/son9ih/pgr/ablation_figures"

ENVS = ["HalfCheetah-v2", "Hopper-v2", "reacher-hard-v0"]
ALPHAS = [0.2, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2]

PATTERN = re.compile(
    r"^(?P<env>.+?)_epoch(?P<epoch>\d+)_seed(?P<seed>\d+)_alpha(?P<alpha>[\d.]+)_\d{8}_\d{6}$"
)

# ── Load data ──
data = {env: {a: [] for a in ALPHAS} for env in ENVS}

for folder in os.listdir(DATA_DIR):
    m = PATTERN.match(folder)
    if m is None:
        continue
    env = m.group("env")
    seed = int(m.group("seed"))
    alpha = float(m.group("alpha"))
    if env not in ENVS or alpha not in ALPHAS or seed not in SEEDS:
        continue
    npy_files = [f for f in os.listdir(os.path.join(DATA_DIR, folder)) if f.endswith(".npy")]
    if not npy_files:
        continue
    arr = np.load(os.path.join(DATA_DIR, folder, npy_files[0]))
    data[env][alpha].append(arr)

# Merge seeds
for env in ENVS:
    for a in ALPHAS:
        if data[env][a]:
            data[env][a] = np.concatenate(data[env][a])
        else:
            data[env][a] = np.array([])

# ── Per-env percentiles for clipping ──
clip_percentiles = [90, 95, 99, 99.9]
clip_vals = {}
for env in ENVS:
    all_env = np.concatenate([data[env][a] for a in ALPHAS if len(data[env][a]) > 0])
    clip_vals[env] = {p: np.percentile(all_env, p) for p in clip_percentiles}

# ── Colors ──
colors = plt.cm.tab10(np.linspace(0, 0.5, len(ALPHAS)))

# ── KDE: subsample to keep computation fast ──
KDE_SUBSAMPLE = 10000
N_GRID = 500

fig, axes = plt.subplots(len(clip_percentiles), len(ENVS), figsize=(18, 4 * len(clip_percentiles)))

for row, pct in enumerate(clip_percentiles):
    for col, env in enumerate(ENVS):
        ax = axes[row, col]
        xmax = clip_vals[env][pct]
        xs = np.linspace(0, xmax, N_GRID)

        for alpha, color in zip(ALPHAS, colors):
            vals = data[env][alpha]
            if len(vals) == 0:
                continue
            # subsample for KDE speed
            if len(vals) > KDE_SUBSAMPLE:
                sub = np.random.default_rng(42).choice(vals, KDE_SUBSAMPLE, replace=False)
            else:
                sub = vals
            kde = gaussian_kde(sub, bw_method="scott")
            density = kde(xs)
            ax.plot(xs, density, color=color, linewidth=1.5, label=f"a={alpha}")
            ax.fill_between(xs, density, color=color, alpha=0.15)

        ax.set_xlim(left=0, right=xmax)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Reward Value")
        ax.set_ylabel("Density")
        if row == 0:
            ax.set_title(env, fontsize=13, fontweight="bold")
        ax.annotate(
            f"x <= p{pct} ({xmax:.5f})",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        if col == 0:
            ax.set_ylabel(f"Density  (clip p{pct})", fontsize=10)
        ax.legend(fontsize=7, loc="center right")

fig.suptitle("Reward KDE by Env & Alpha  (3 seeds merged, clipped at various percentiles)",
             fontsize=14, y=1.01)
plt.tight_layout()
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, "reward_kde_multi_clip.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {save_path}")
