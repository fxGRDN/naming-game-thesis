import numpy as np
import matplotlib.pyplot as plt

stats = np.load("base_game_monte_carlo_stats.npy")



x = np.arange(stats.shape[2])

plt.figure(figsize=(15, 20))

for i in range(stats.shape[0]):
    plt.subplot(5, 2, i + 1)
    plt.plot(x, stats[i].mean(axis=-2), alpha=0.1, color="blue")
    plt.title(f"Memory Size: {(i+1)*10}")



plt.savefig("monte_carlo_base_game_stats.png")
