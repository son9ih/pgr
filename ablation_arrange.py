import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_path = "/home/son9ih/pgr/ablation_data/reacher-hard-v0_epoch9_seed5_alpha2.0_20260416_202320/reward_vector_epoch9.npy"
reward = np.load(data_path)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(reward, bins=100, edgecolor="black", alpha=0.7)
ax.set_xlabel("Reward")
ax.set_ylabel("Count")
ax.set_title(f"Reward Histogram (reacher-hard-v0, epoch=9, seed=5, alpha=2.0)\n"
             f"n={len(reward)}, mean={reward.mean():.6f}, std={reward.std():.6f}")
ax.axvline(reward.mean(), color="red", linestyle="--", label=f"mean={reward.mean():.6f}")
ax.legend()
plt.tight_layout()

save_path = "/home/son9ih/pgr/ablation_figures/reward_hist_reacher-hard-v0_epoch9_seed5_alpha2.0.png"
plt.savefig(save_path, dpi=150)
plt.close(fig)
print(f"Saved histogram to {save_path}")
