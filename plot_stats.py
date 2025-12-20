import numpy as np
import matplotlib.pyplot as plt

stats = np.load("base_game_monte_carlo_stats.npy")
# stats shape: (metrics=3, steps, iters)

x = np.arange(stats.shape[-1])  # steps

xd = stats.transpose(1, 2, 0)

def mean_q(metric_idx):
    data = xd[metric_idx]  # shape (steps, iters)
    mean = data.mean(axis=-1)
    lo = np.percentile(data, 2.5, axis=1)
    hi = np.percentile(data, 97.5, axis=1)
    return mean, lo, hi

plt.figure(figsize=(12, 4))

# Success Rate
m, lo, hi = mean_q(0)
plt.subplot(1, 3, 1)
plt.plot(x, m, color="C0")
plt.fill_between(x, lo, hi, color="C0", alpha=0.2)
plt.title("Success Rate")
plt.xlabel("Rounds")
plt.ylim(0, 1.2)

# Coherence
m, lo, hi = mean_q(1)
plt.subplot(1, 3, 2)
plt.plot(x, m, color="C1")
plt.fill_between(x, lo, hi, color="C1", alpha=0.2)
plt.title("Coherence")
plt.xlabel("Rounds")
plt.ylim(0, 1.2)

# Unique Words
m, lo, hi = mean_q(2)
plt.subplot(1, 3, 3)
plt.plot(x, m, color="C2")
plt.fill_between(x, lo, hi, color="C2", alpha=0.2)
plt.title("Unique Words")
plt.xlabel("Rounds")

plt.tight_layout()
plt.savefig("plots/monte_carlo_base_game_stats.png")



